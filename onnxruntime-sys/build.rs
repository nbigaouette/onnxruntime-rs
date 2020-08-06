use std::{env, path::PathBuf};

fn main() {
    let onnxruntime_install_dir: PathBuf = env::var("ONNXRUNTIME_INSTALL_DIR").unwrap().into();
    let include_dir = onnxruntime_install_dir.join("include/onnxruntime/core/session");
    let clang_arg = format!("-I{}", include_dir.display());

    // Tell cargo to tell rustc to link the system onnxruntime
    // shared library.
    println!("cargo:rustc-link-lib=onnxruntime");
    println!(
        "cargo:rustc-link-search={}/lib",
        onnxruntime_install_dir.display()
    );

    // Tell cargo to invalidate the built crate whenever the wrapper changes
    println!("cargo:rerun-if-changed=wrapper.h");

    // The bindgen::Builder is the main entry point
    // to bindgen, and lets you build up options for
    // the resulting bindings.
    let bindings = bindgen::Builder::default()
        // The input header we would like to generate
        // bindings for.
        .header("wrapper.h")
        // The current working directory is 'onnxruntime-sys'
        .clang_arg(clang_arg)
        // Tell cargo to invalidate the built crate whenever any of the
        // included header files changed.
        .parse_callbacks(Box::new(bindgen::CargoCallbacks))
        // Finish the builder and generate the bindings.
        .generate()
        // Unwrap the Result and panic on failure.
        .expect("Unable to generate bindings");

    // Write the bindings to the $OUT_DIR/bindings.rs file.
    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    bindings
        .write_to_file(out_path.join("bindings.rs"))
        .expect("Couldn't write bindings!");
}
