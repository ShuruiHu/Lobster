#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/gather.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/set_operations.h>

#include <cstddef>
#include <type_traits>
#include <unordered_map>

#include "dataflow.h"
#include "device_vec.h"
#include "expr.h"
#include "filter.h"
#include "find.h"
#include "flame.h"
#include "intersect.h"
#include "join.h"
#include "merge.h"
#include "normalize.h"
#include "overwrite_one.h"
#include "product.h"
#include "project.h"
#include "provenance.h"
#include "table_index.h"
#include "utils.h"

namespace {
using ProofProvenance = DiffTopKProofsProvenance<>;

// Normalization sorts U32/F32 columns by their uint32_t representation,
// with the sample index before the tuple columns.
__device__ int compare_proof_rows(
    const TableView<ProofProvenance> &left, const char *left_samples,
    size_t left_row, const TableView<ProofProvenance> &right,
    const char *right_samples, size_t right_row, size_t width) {
  if (left_samples[left_row] != right_samples[right_row]) {
    return left_samples[left_row] < right_samples[right_row] ? -1 : 1;
  }
  for (size_t col = 0; col < width; col++) {
    auto l = left.at<uint32_t>(left_row, col);
    auto r = right.at<uint32_t>(right_row, col);
    if (l != r) return l < r ? -1 : 1;
  }
  return 0;
}

__global__ void select_proof_frontier(
    TableView<ProofProvenance> stable, const char *stable_samples,
    const ProofProvenance::Tag *stable_tags, size_t stable_size,
    TableView<ProofProvenance> candidates, const char *candidate_samples,
    const ProofProvenance::Tag *candidate_tags, size_t width,
    ProofProvenance::BatchDeviceContext contexts, char *stable_keep,
    char *candidate_keep) {
  size_t row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= candidates.size()) return;
  candidate_keep[row] = 0;
  const auto candidate = candidate_tags[row];
  if (candidate.empty_) return;

  size_t lo = 0, hi = stable_size;
  while (lo < hi) {
    auto mid = lo + (hi - lo) / 2;
    if (compare_proof_rows(stable, stable_samples, mid, candidates,
                           candidate_samples, row, width) < 0) {
      lo = mid + 1;
    } else {
      hi = mid;
    }
  }
  bool found = lo < stable_size &&
      compare_proof_rows(stable, stable_samples, lo, candidates,
                         candidate_samples, row, width) == 0;
  if (found && !stable_tags[lo].empty_) {
    auto context =
        ProofProvenance::sample_context(contexts, candidate_samples[row]);
    // CPU scheduling adds (stable, candidate), retaining stable on a tie.
    if (!(ProofProvenance::clause_probability(context, candidate) >
          ProofProvenance::clause_probability(context, stable_tags[lo]))) {
      return;
    }
  }
  candidate_keep[row] = 1;
  // Normalized candidates have unique keys, so each stable mask has one writer.
  if (found) stable_keep[lo] = 0;
}

struct KeepProofRow {
  const char *keep;
  __device__ bool operator()(uint32_t row) const { return keep[row] != 0; }
};

Table<ProofProvenance> compact_proof_rows(
    const Table<ProofProvenance> &source, const device_vec<char> &keep,
    const Allocator &alloc) {
  if (source.size() == 0) return source;
  device_vec<uint32_t> indices(source.size(), alloc);
  auto rows = thrust::make_counting_iterator<uint32_t>(0);
  auto end = thrust::copy_if(thrust::device, rows, rows + source.size(),
                             indices.data(), KeepProofRow{keep.data()});
  size_t size = end - indices.data();
  if (size == source.size()) return source;
  if (size == 0) return Table<ProofProvenance>(source.schema());

  Array<device_buffer> facts(source.width());
  for (size_t col = 0; col < source.width(); col++) {
    new (&facts[col]) device_buffer(size,
        source.schema().flatten().at(col).singleton(), alloc);
    thrust::gather(thrust::device, indices.data(), end,
        reinterpret_cast<const uint32_t *>(source.at_raw(0, col)),
        facts[col].begin<uint32_t>());
  }
  device_vec<ProofProvenance::Tag> tags(size, alloc);
  thrust::gather(thrust::device, indices.data(), end,
                 source.tags().data(), tags.data());
  device_vec<char> samples(size, alloc);
  thrust::gather(thrust::device, indices.data(), end,
                 source.sample_mask().data(), samples.data());
  return Table<ProofProvenance>(source.schema(), std::move(tags),
                                std::move(facts), std::move(samples));
}

bool advance_proof_frontier(DynamicRelation<ProofProvenance> &relation,
                            const ProofProvenance &ctx, const Allocator &alloc) {
  auto candidates = normalized(relation.delta(), ctx, alloc);
  if (candidates.size() == 0) {
    relation.recent() = candidates;
    return false;
  }
  auto width = candidates.width();
  if (width == 0 || width > 6) {
    PANIC("Unsupported proof frontier width: %lu", width);
  }
  for (size_t col = 0; col < width; col++) {
    auto type = candidates.schema().flatten().at(col).singleton();
    if (type != ValueType::U32() && type != ValueType::F32()) {
      PANIC("Unsupported proof frontier column type");
    }
  }
  auto &stable = relation.stable();
  device_vec<char> stable_keep(stable.size(), alloc);
  device_vec<char> candidate_keep(candidates.size(), alloc);
  if (stable.size() != 0) {
    cudaCheck(cudaMemset(stable_keep.data(), 1, stable.size()));
  }
  // An empty Table has no storage. The dummy view is never read when size is zero.
  auto stable_view = stable.size() != 0 ? stable.view() : candidates.view();
  auto stable_samples = stable.size() != 0 ? stable.sample_mask().data() : nullptr;
  auto stable_tags = stable.size() != 0 ? stable.tags().data() : nullptr;
  select_proof_frontier<<<ROUND_UP_TO_NEAREST(candidates.size(), 128), 128>>>(
      stable_view, stable_samples, stable_tags, stable.size(),
      candidates.view(), candidates.sample_mask().data(), candidates.tags().data(),
      width, ctx.device_context(), stable_keep.data(), candidate_keep.data());
  cudaCheck(cudaDeviceSynchronize());
  relation.stable() = compact_proof_rows(stable, stable_keep, alloc);
  relation.recent() = compact_proof_rows(candidates, candidate_keep, alloc);
  return relation.recent().size() != 0;
}
}  // namespace

template <typename Prov>
std::ostream &operator<<(std::ostream &os,
                         const DynamicRelation<Prov> &relation) {
  os << "{\n\tstable:\n"
     << relation.stable() << "\n\trecent:\n"
     << relation.recent() << "\n\tdelta:\n"
     << relation.delta() << "}" << std::endl;
  return os;
}

template <typename Prov>
DynamicIdb<Prov>::DynamicIdb() {}
template <typename Prov>
DynamicIdb<Prov>::DynamicIdb(
    std::unordered_map<std::string, DynamicRelation<Prov>> &&relations)
    : relations_(std::move(relations)) {}

template <typename Prov>
std::vector<std::string> DynamicIdb<Prov>::incorporate_delta(const Prov &ctx, Allocator alloc) {
  TRACE_START(DynamicIdb_incorporate_delta);
  std::vector<std::string> changed_relations;
  this->validate();

  for (auto &&pair : relations_) {
    hINFO("Incorporating delta for table " << pair.first);

    auto &relation = pair.second;

    auto stable_size = relation.stable().size();

    relation.stable() =
        merge_tables(relation.stable(), relation.recent(), ctx, alloc);
    this->validate();
    if constexpr (std::is_same_v<Prov, ProofProvenance>) {
      if (advance_proof_frontier(relation, ctx, alloc)) {
        changed_relations.push_back(pair.first);
      }
      relation.delta().clear();
      this->validate();
      continue;
    }
    if (stable_size != relation.stable().size()) {
      changed_relations.push_back(pair.first);
    }

    if (Prov::is_unit) {
      relation.recent() = normalized(relation.delta(), ctx, alloc);
      // perform set difference to subtract stable from recent
      if (relation.schema().flatten() == TupleType({ValueType::U32(), ValueType::U32()})) {
        TRACE_START(DynamicIdb_incorporate_delta_difference);
        if (relation.stable().size() == 0) {
          hINFO("Stable relation is empty, skipping difference");
          continue;
        }
        if (relation.recent().size() == 0) {
          hINFO("Recent relation is empty, skipping difference");
          continue;
        }

        hINFO("Difference input left:");
        hINFO(relation.recent());
        hINFO("Difference input right:");
        hINFO(relation.stable());

        device_vec<char> output_sample_mask(relation.recent().size(), alloc);
        Array<device_buffer> output_facts(2);
        new (&output_facts[0]) device_buffer(relation.recent().size(), ValueType::U32(), alloc);
        new (&output_facts[1]) device_buffer(relation.recent().size(), ValueType::U32(), alloc);
        auto recent_in = thrust::make_zip_iterator(
            thrust::make_tuple(relation.recent().sample_mask().data(),
                               relation.recent().values()[0].template cbegin<uint32_t>().data(),
                               relation.recent().values()[1].template cbegin<uint32_t>().data()));
        auto stable_in = thrust::make_zip_iterator(
            thrust::make_tuple(relation.stable().sample_mask().data(),
                               relation.stable().values()[0].template cbegin<uint32_t>().data(),
                               relation.stable().values()[1].template cbegin<uint32_t>().data()));
        auto output = thrust::make_zip_iterator(
            thrust::make_tuple(
              output_sample_mask.data(),
              output_facts[0].template begin<uint32_t>().data(),
              output_facts[1].template begin<uint32_t>().data()));

        auto new_end = thrust::set_difference(
            thrust::device,
            recent_in, recent_in + relation.recent().size(),
            stable_in, stable_in + relation.stable().size(),
            output);
        auto new_size = thrust::distance(output, new_end);
        output_sample_mask.resize(new_size);
        output_facts[0].resize(new_size);
        output_facts[1].resize(new_size);

        device_vec<typename Prov::Tag> output_tags;
        relation.recent() = Table<Prov>(relation.schema(), std::move(output_tags),
                   std::move(output_facts), std::move(output_sample_mask));
        hINFO("Difference output:");
        hINFO(relation.recent());
      }
    } else {
      relation.recent() = normalized(relation.delta(), ctx, alloc);
    }

    this->validate();

    relation.delta().clear();
  }
  this->validate();

  return changed_relations;
}

template <typename Prov>
void DynamicIdb<Prov>::validate() const {
  if (!SAFETY) {
    return;
  }
  for (auto &&pair : relations_) {
    auto &relation = pair.second;
    relation.stable().validate();
    relation.recent().validate();
    relation.delta().validate();

    auto stable_sorted_until = relation.stable().sorted_until();
    if (stable_sorted_until != relation.stable().size()) {
      std::cout << "Error: stable relation for " << pair.first
                << " only sorted until " << stable_sorted_until
                << " but length is " << relation.stable().size() << std::endl;
      PANIC("Stable relation not sorted. See prior logging for more.");
    }
    auto recent_sorted_until = relation.recent().sorted_until();
    if (recent_sorted_until != relation.recent().size()) {
      std::cout << "Error: recent relation for " << pair.first
                << " only sorted until " << recent_sorted_until
                << " but length is " << relation.recent().size() << std::endl;
      PANIC("Recent relation not sorted. See prior logging for more.");
    }
  }
}

template <typename Prov>
const std::unordered_map<std::string, DynamicRelation<Prov>> &
DynamicIdb<Prov>::relations() const {
  return relations_;
}
template <typename Prov>
std::unordered_map<std::string, DynamicRelation<Prov>> &
DynamicIdb<Prov>::relations() {
  return relations_;
}

template <typename Prov>
const DynamicRelation<Prov> &DynamicIdb<Prov>::operator[](
    const std::string &name) const {
  if (relations_.find(name) == relations_.end()) {
    std::string all_relations;
    for (const auto &pair : relations_) {
      all_relations += pair.first;
      all_relations += "\n";
    }
    throw std::runtime_error(std::string("DynamicIdb: unknown relation ") +
                             name + std::string("\nValid relations are\n") +
                             all_relations);
  }
  return relations_.at(name);
}
template <typename Prov>
DynamicRelation<Prov> &DynamicIdb<Prov>::operator[](const std::string &name) {
  if (relations_.find(name) == relations_.end()) {
    std::string all_relations;
    for (const auto &pair : relations_) {
      all_relations += pair.first;
    }
    throw std::runtime_error(std::string("DynamicIdb: unknown relation ") +
                             name + std::string("\nValid relations are ") +
                             all_relations);
  }
  return relations_.at(name);
}

template <typename Prov>
std::ostream &operator<<(std::ostream &os, const DynamicIdb<Prov> &idb) {
  os << "DynamicIdb{\n";
  for (auto &pair : idb.relations_) {
    os << pair.first << ":\n" << pair.second << "\n";
  }
  os << "}" << std::endl;
  return os;
}

template <typename Prov>
std::unique_ptr<ActiveDataflow<Prov>> ActiveDataflow<Prov>::from_dataflow(
    const Dataflow &df, DynamicIdb<Prov> &idb) {
  if (df.tag == Dataflow::Tag::Relation) {
    auto name = df.relation._0.to_string();
    return std::make_unique<ActiveDataflowSource<Prov>>(
        ActiveDataflowSource<Prov>::Op::Relation, name, idb[name].schema());
  } else if (df.tag == Dataflow::Tag::Project) {
    return std::make_unique<ActiveDataflowUnaryOp<Prov>>(
        ActiveDataflowUnaryOp<Prov>::Op::Project,
        std::move(from_dataflow(*df.project._0, idb)), df.project._1);
  } else if (df.tag == Dataflow::Tag::Join) {
    auto result = std::make_unique<ActiveDataflowBinaryOp<Prov>>(
        ActiveDataflowBinaryOp<Prov>::Op::Join,
        std::move(from_dataflow(*df.join._0, idb)),
        std::move(from_dataflow(*df.join._1, idb)));
    result->index_on_right() = df.join.index_on_right;
    return result;
  } else if (df.tag == Dataflow::Tag::Product) {
    return std::make_unique<ActiveDataflowBinaryOp<Prov>>(
        ActiveDataflowBinaryOp<Prov>::Op::Product,
        std::move(from_dataflow(*df.product._0, idb)),
        std::move(from_dataflow(*df.product._1, idb)));
  } else if (df.tag == Dataflow::Tag::Intersect) {
    auto result = std::make_unique<ActiveDataflowBinaryOp<Prov>>(
        ActiveDataflowBinaryOp<Prov>::Op::Intersection,
        std::move(from_dataflow(*df.intersect._0, idb)),
        std::move(from_dataflow(*df.intersect._1, idb)));
    result->index_on_right() = false;
    return result;
  } else if (df.tag == Dataflow::Tag::Filter) {
    return std::make_unique<ActiveDataflowUnaryOp<Prov>>(
        ActiveDataflowUnaryOp<Prov>::Op::Filter,
        std::move(from_dataflow(*df.filter._0, idb)), df.filter._1);
  } else if (df.tag == Dataflow::Tag::Find) {
    return std::make_unique<ActiveDataflowUnaryOp<Prov>>(
        ActiveDataflowUnaryOp<Prov>::Op::Find,
        std::move(from_dataflow(*df.find._0, idb)), Expr(df.find._1));
  } else if (df.tag == Dataflow::Tag::OverwriteOne) {
    Array<Expr> dummy_expr;
    return std::make_unique<ActiveDataflowUnaryOp<Prov>>(
        ActiveDataflowUnaryOp<Prov>::Op::OverwriteOne,
        std::move(from_dataflow(*df.overwrite_one._0, idb)), Expr(dummy_expr));
  } else {
    throw std::runtime_error("ActiveDataflow: unimplemented");
  }
}

template <typename Prov>
ActiveDataflowSource<Prov>::ActiveDataflowSource(Op op, std::string name,
                                                 TupleType schema)
    : op_(op), name_(name), schema_(schema) {}

template <typename Prov>
TupleType ActiveDataflowSource<Prov>::result_schema() const {
  return schema_;
}
template <typename Prov>
std::vector<std::string> ActiveDataflowSource<Prov>::dependencies() const {
  return {name_};
}

template <typename Prov>
Table<Prov> ActiveDataflowSource<Prov>::evaluate_stable(
    const DynamicIdb<Prov> &idb, const Prov &) const {
  if (op_ == Op::Relation) {
    return idb[name_].stable();
  } else {
    throw std::runtime_error("ActiveDataflowSource: unknown op");
  }
}
template <typename Prov>
Table<Prov> ActiveDataflowSource<Prov>::evaluate_recent(
    const DynamicIdb<Prov> &idb, const Prov &) const {
  if (op_ == Op::Relation) {
    return idb[name_].recent();
  } else {
    throw std::runtime_error("ActiveDataflowSource: unknown op");
  }
}

template <typename Prov>
void ActiveDataflowSource<Prov>::serialize(std::ostream &os) const {
  std::string op;
  switch (op_) {
    case Op::Relation:
      op = "Relation";
      break;
  }
  os << "ActiveDataflowSource{" << std::endl
     << "op: " << op << std::endl
     << "name: " << name_ << std::endl
     << "schema: " << schema_ << std::endl
     << "}";
}

template <typename Prov>
ActiveDataflowBinaryOp<Prov>::ActiveDataflowBinaryOp(
    Op op, std::unique_ptr<ActiveDataflow<Prov>> left,
    std::unique_ptr<ActiveDataflow<Prov>> right)
    : op_(op),
      index_on_right_(false),
      left_(std::move(left)),
      right_(std::move(right)) {}

template <typename Prov>
TupleType ActiveDataflowBinaryOp<Prov>::result_schema() const {
  if (op_ == Op::Join) {
    auto [left_first, left_rest] = left_->result_schema().remove_first();
    auto [right_first, right_rest] = right_->result_schema().remove_first();
    if (left_first != right_first) {
      throw std::runtime_error(
          "ActiveDataflowBinaryOp: join of tables with "
          "different index column schemas");
    }
    TupleType schema{std::vector<TupleType>{left_first}};
    schema = schema.append(left_rest);
    schema = schema.append(right_rest);
    return schema;
  } else if (op_ == Op::Product) {
    TupleType schema{std::vector<TupleType>{left_->result_schema(),
                                            right_->result_schema()}};
    return schema;
  } else {
    return left_->result_schema();
  }
}

template <typename Prov>
std::vector<std::string> ActiveDataflowBinaryOp<Prov>::dependencies() const {
  auto left_deps = left_->dependencies();
  auto right_deps = right_->dependencies();
  left_deps.insert(left_deps.end(), right_deps.begin(), right_deps.end());
  return left_deps;
}

template <typename Prov>
Table<Prov> ActiveDataflowBinaryOp<Prov>::evaluate_stable(
    const DynamicIdb<Prov> &idb, const Prov &ctx) const {
  if (TRACE_FINE()) {
    TRACE_START(ActiveDataflowBinaryOp_evaluate_stable);
  }
  const auto left = left_->evaluate_stable(idb, ctx);
  const auto right = right_->evaluate_stable(idb, ctx);
  if (op_ == Op::Join) {
    hINFO("SCHEMA: JOIN: LEFT: " << left.schema());
    hINFO("SCHEMA: JOIN: RIGHT: " << right.schema());

    return join(left, right, result_schema(), ctx, this->index_on_right_);
  } else if (op_ == Op::Product) {
    return product(left, right, result_schema(), ctx);
  } else if (op_ == Op::Difference) {
    throw std::runtime_error(
        "ActiveDataflowBinaryOp: difference not implemented");
  } else if (op_ == Op::Intersection) {
    return intersect(left, right, result_schema(), ctx, this->index_on_right_);
  } else if (op_ == Op::Union) {
    throw std::runtime_error("ActiveDataflowBinaryOp: union not implemented");
  } else {
    throw std::runtime_error("ActiveDataflowBinaryOp: unknown op");
  }
}

template <typename Prov>
Table<Prov> ActiveDataflowBinaryOp<Prov>::evaluate_recent(
    const DynamicIdb<Prov> &idb, const Prov &ctx) const {
  if (TRACE_FINE()) {
    TRACE_START(ActiveDataflowBinaryOp_evaluate_recent);
  }
  const auto left_stable = left_->evaluate_stable(idb, ctx);
  const auto left_recent = left_->evaluate_recent(idb, ctx);
  const auto right_stable = right_->evaluate_stable(idb, ctx);
  const auto right_recent = right_->evaluate_recent(idb, ctx);

  if (op_ == Op::Join) {
    hINFO("SCHEMA: JOIN: LEFT: " << left_recent.schema());
    hINFO("SCHEMA: JOIN: RIGHT: " << right_recent.schema());
    auto stable_recent = join(left_stable, right_recent, result_schema(), ctx,
                              this->index_on_right_);
    auto recent_stable = join(left_recent, right_stable, result_schema(), ctx,
                              this->index_on_right_);
    auto recent_recent = join(left_recent, right_recent, result_schema(), ctx,
                              this->index_on_right_);
    stable_recent.append(recent_stable);
    stable_recent.append(recent_recent);
    return stable_recent;
  } else if (op_ == Op::Product) {
    auto stable_recent =
        product(left_stable, right_recent, result_schema(), ctx);
    auto recent_stable =
        product(left_recent, right_stable, result_schema(), ctx);
    auto recent_recent =
        product(left_recent, right_recent, result_schema(), ctx);
    stable_recent.append(recent_stable);
    stable_recent.append(recent_recent);
    return stable_recent;
  } else if (op_ == Op::Difference) {
    throw std::runtime_error(
        "ActiveDataflowBinaryOp: difference not implemented");
  } else if (op_ == Op::Intersection) {
    auto stable_recent = intersect(left_stable, right_recent, result_schema(),
                                   ctx, this->index_on_right_);
    auto recent_stable = intersect(left_recent, right_stable, result_schema(),
                                   ctx, this->index_on_right_);
    auto recent_recent = intersect(left_recent, right_recent, result_schema(),
                                   ctx, this->index_on_right_);
    stable_recent.append(recent_stable);
    stable_recent.append(recent_recent);
    return stable_recent;
  } else if (op_ == Op::Union) {
    throw std::runtime_error("ActiveDataflowBinaryOp: union not implemented");
  } else {
    throw std::runtime_error("ActiveDataflowBinaryOp: unknown op");
  }
}

template <typename Prov>
bool &ActiveDataflowBinaryOp<Prov>::index_on_right() {
  return index_on_right_;
}

template <typename Prov>
void ActiveDataflowBinaryOp<Prov>::serialize(std::ostream &os) const {
  std::string op;
  switch (op_) {
    case Op::Join:
      op = "Join";
      break;
    case Op::Product:
      op = "Product";
      break;
    case Op::Difference:
      op = "Difference";
      break;
    case Op::Intersection:
      op = "Intersection";
      break;
    case Op::Union:
      op = "Union";
      break;
  }
  os << "ActiveDataflowBinaryOp{" << std::endl
     << "result: " << result_schema() << std::endl
     << "op: " << op << std::endl
     << "left: " << *left_ << std::endl
     << "right: " << *right_ << std::endl
     << "}";
}

template <typename Prov>
ActiveDataflowUnaryOp<Prov>::ActiveDataflowUnaryOp(
    Op op, std::unique_ptr<ActiveDataflow<Prov>> source, Expr expr)
    : op_(op), source_(std::move(source)), expr_(expr) {}

template <typename Prov>
TupleType ActiveDataflowUnaryOp<Prov>::result_schema() const {
  if (op_ == Op::Project) {
    return expr_.result_type(source_->result_schema());
  } else if (op_ == Op::Filter) {
    return source_->result_schema();
  } else if (op_ == Op::Find) {
    return source_->result_schema();
  } else if (op_ == Op::OverwriteOne) {
    return source_->result_schema();
  } else {
    return source_->result_schema();
  }
}

template <typename Prov>
std::vector<std::string> ActiveDataflowUnaryOp<Prov>::dependencies() const {
  return source_->dependencies();
}

template <typename Prov>
Table<Prov> ActiveDataflowUnaryOp<Prov>::evaluate_stable(
    const DynamicIdb<Prov> &idb, const Prov &ctx) const {
  if (TRACE_FINE()) {
    TRACE_START(ActiveDataflowUnaryOp_evaluate_stable);
  }
  auto source = source_->evaluate_stable(idb, ctx);
  if (op_ == Op::Project) {
    return project(source, expr_, result_schema(), ctx);
  } else if (op_ == Op::Filter) {
    return filter(source, expr_, result_schema());
  } else if (op_ == Op::Find) {
    if (expr_.tag != Expr::Tag::Constant) {
      PANIC("ActiveDataflowUnaryOp: find with non-value");
    }
    return find(source, expr_.constant._0, result_schema());
  } else if (op_ == Op::OverwriteOne) {
    return overwrite_one(source, result_schema());
  } else {
    throw std::runtime_error("ActiveDataflowUnaryOp: unknown op");
  }
}

template <typename Prov>
Table<Prov> ActiveDataflowUnaryOp<Prov>::evaluate_recent(
    const DynamicIdb<Prov> &idb, const Prov &ctx) const {
  if (TRACE_FINE()) {
    TRACE_START(ActiveDataflowUnaryOp_evaluate_recent);
  }
  auto source = source_->evaluate_recent(idb, ctx);
  if (op_ == Op::Project) {
    auto result = project(source, expr_, result_schema(), ctx);
    if (TRACE_FINE()) {
      TRACE_START(ActiveDataflowUnaryOp_evaluate_recent_project);
    }
    return result;
  } else if (op_ == Op::Filter) {
    return filter(source, expr_, result_schema());
  } else if (op_ == Op::Find) {
    if (expr_.tag != Expr::Tag::Constant) {
      PANIC("ActiveDataflowUnaryOp: find with non-value");
    }
    return find(source, expr_.constant._0, result_schema());
  } else if (op_ == Op::OverwriteOne) {
    return overwrite_one(source, result_schema());
  } else {
    throw std::runtime_error("ActiveDataflowUnaryOp: unknown op");
  }
}

template <typename Prov>
void ActiveDataflowUnaryOp<Prov>::serialize(std::ostream &os) const {
  std::string op;
  switch (op_) {
    case Op::Project:
      op = "Project";
      break;
    case Op::Filter:
      op = "Filter";
      break;
    case Op::Find:
      op = "Find";
      break;
    case Op::OverwriteOne:
      op = "OverwriteOne";
      break;
  }
  os << "ActiveDataflowUnaryOp{" << std::endl
     << "result: " << result_schema() << std::endl
     << "op: " << op << std::endl
     << "expr: " << expr_ << std::endl
     << "source: " << *source_ << std::endl
     << "}";
}

#define PROV UnitProvenance
template class DynamicIdb<PROV>;
template std::ostream &operator<<(std::ostream &,
                                  const DynamicIdb<PROV> &);
template class ActiveDataflow<PROV>;
template class ActiveDataflowSource<PROV>;
template class ActiveDataflowBinaryOp<PROV>;
template class ActiveDataflowUnaryOp<PROV>;
#undef PROV
#define PROV MinMaxProbProvenance
template class DynamicIdb<PROV>;
template std::ostream &operator<<(std::ostream &,
                                  const DynamicIdb<PROV> &);
template class ActiveDataflow<PROV>;
template class ActiveDataflowSource<PROV>;
template class ActiveDataflowBinaryOp<PROV>;
template class ActiveDataflowUnaryOp<PROV>;
#undef PROV
#define PROV DiffMinMaxProbProvenance
template class DynamicIdb<PROV>;
template std::ostream &operator<<(std::ostream &,
                                  const DynamicIdb<PROV> &);
template class ActiveDataflow<PROV>;
template class ActiveDataflowSource<PROV>;
template class ActiveDataflowBinaryOp<PROV>;
template class ActiveDataflowUnaryOp<PROV>;
#undef PROV
#define PROV DiffAddMultProbProvenance<>
template class DynamicIdb<PROV>;
template std::ostream &operator<<(std::ostream &,
                                  const DynamicIdb<PROV> &);
template class ActiveDataflow<PROV>;
template class ActiveDataflowSource<PROV>;
template class ActiveDataflowBinaryOp<PROV>;
template class ActiveDataflowUnaryOp<PROV>;
#undef PROV
#define PROV DiffTopKProofsProvenance<>
template class DynamicIdb<PROV>;
template std::ostream &operator<<(std::ostream &,
                                  const DynamicIdb<PROV> &);
template class ActiveDataflow<PROV>;
template class ActiveDataflowSource<PROV>;
template class ActiveDataflowBinaryOp<PROV>;
template class ActiveDataflowUnaryOp<PROV>;
#undef PROV
