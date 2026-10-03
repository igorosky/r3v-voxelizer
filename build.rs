#[cfg(feature = "c")]
fn gen_c() {
    let crate_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    cbindgen::Builder::new()
        .with_crate(crate_dir)
        .with_language(cbindgen::Language::C)
        .with_braces(cbindgen::Braces::SameLine)
        .with_include_guard("R3V_VOXELIZER_GUARD")
        .generate()
        .expect("Failed to generate C bindings")
        .write_to_file("bindings/r3v-voxelizer.h");
}

#[cfg(feature = "cpp")]
fn gen_cpp() {
    let crate_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    cbindgen::Builder::new()
        .with_crate(crate_dir)
        .with_language(cbindgen::Language::Cxx)
        .with_braces(cbindgen::Braces::SameLine)
        .with_pragma_once(true)
        .with_namespaces(&["r3v", "voxelizer"])
        .generate()
        .expect("Failed to generate C bindings")
        .write_to_file("bindings/r3v-voxelizer.hpp");
}

fn main() {
    #[cfg(feature = "c")]
    gen_c();
    #[cfg(feature = "cpp")]
    gen_cpp();
}
