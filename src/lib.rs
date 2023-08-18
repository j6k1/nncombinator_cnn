//! nncombinator_cnn is a neural network library that allows type-safe implementation.

#![allow(incomplete_features)]
#![feature(generic_const_exprs)]

extern crate libc;
extern crate cuda_runtime_sys;
extern crate rcublas_sys;
extern crate rcublas;
extern crate rcudnn;
extern crate rcudnn_sys;
extern crate const_guards;
extern crate rayon;

extern crate nncombinator;

pub mod collection;
pub mod device;
