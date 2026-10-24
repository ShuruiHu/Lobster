use std::{env, path::PathBuf};

fn main() {
  // The fixed source contains grammar.lalrpop but no generated grammar.rs.
  lalrpop::Configuration::new().generate_in_source_tree().process().unwrap();
  println!("cargo:rerun-if-changed=src/compiler/front/grammar.lalrpop");
  let manifest = PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").unwrap());
  let profile = env::var("PROFILE").unwrap();
  let build_dir = env::var_os("SCLGPU_LIB_DIR").map(PathBuf::from).unwrap_or_else(|| {
    manifest.join("src/gpu_runtime/libsclgpu").join(if profile == "release" { "cmake-release" } else { "cmake-debug" })
  });
  let archive = build_dir.join("libsclgpu.a");
  assert!(archive.is_file(), "Build libsclgpu first: {}", archive.display());
  println!("cargo:rerun-if-env-changed=SCLGPU_LIB_DIR");
  println!("cargo:rerun-if-env-changed=CUDA_HOME");
  println!("cargo:rerun-if-changed={}", archive.display());
  println!("cargo:rustc-link-search=native={}", build_dir.display());
  println!("cargo:rustc-link-lib=static=sclgpu");
  println!("cargo:rustc-link-lib=stdc++");
  let cuda = PathBuf::from(env::var_os("CUDA_HOME").unwrap_or_else(|| "/usr/local/cuda".into()));
  println!("cargo:rustc-link-search=native={}", cuda.join("targets/x86_64-linux/lib").display());
  println!("cargo:rustc-link-lib=cudart_static");
  println!("cargo:rustc-link-lib=cudadevrt");
}
