#![allow(incomplete_features)]
#![feature(generic_const_exprs)]

extern crate nncombinator_cnn;

pub mod common;

use std::cell::RefCell;
use std::fs;
use std::ops::DerefMut;
use std::path::{Path, PathBuf};
use std::rc::Rc;
use nncombinator::activation::{ReLu, SoftMax};
use nncombinator::arr::Arr;
use nncombinator::device::DeviceCpu;
use nncombinator::layer::activation::ActivationLayer;
use nncombinator::layer::{AddLayer, AddLayerTrain, BatchForward, BatchTrain, ForwardAll};
use nncombinator::layer::bridge::{BridgeLayerBuilder};
use nncombinator::layer::input::InputLayer;
use nncombinator::layer::linear::LinearLayerBuilder;
use nncombinator::layer::output::LinearOutputLayer;
use nncombinator::lossfunction::CrossEntropyMulticlass;
use nncombinator::optimizer::MomentumSGD;
use rand::{prelude, Rng, SeedableRng};
use rand::prelude::{Distribution, SliceRandom};
use rand_distr::Normal;
use rand_xorshift::XorShiftRng;
use nncombinator_cnn::collection::{Images};
use nncombinator_cnn::layer::convolution::ConvolutionLayerBuilder;
use nncombinator_cnn::layer::pooling::MaxPooling2DBuilder;
use crate::common::SHARED_MEMORY_POOL;

#[test]
fn test_mnist() {
    let mut rnd = prelude::thread_rng();
    let rnd_base = Rc::new(RefCell::new(XorShiftRng::from_seed(rnd.gen())));

    let n1 = Normal::<f32>::new(0.0, (2f32/(28f32*28f32)).sqrt()).unwrap();
    let n2 = Normal::<f32>::new(0.0, (2f32/(16f32*14f32*14f32)).sqrt()).unwrap();
    let n3 = Normal::<f32>::new(0.0, (2f32/(32f32*7f32*7f32)).sqrt()).unwrap();
    let n4 = Normal::<f32>::new(0.0, 1f32/(120f32).sqrt()).unwrap();

    let device = DeviceCpu::new().unwrap();

    let net:InputLayer<f32,Images<f32,1,28,28>,_> = InputLayer::new();

    let rnd = rnd_base.clone();

    let mut net = net.add_layer(|l| {
        let rnd = rnd.clone();
        ConvolutionLayerBuilder::<1, 16, 28, 28, 3, 3, 1, 1>::new().build(l, &device, move || n1.sample(&mut rnd.borrow_mut().deref_mut())).unwrap()
    }).add_layer(|l| {
        ActivationLayer::new(l,ReLu::new(&device),&device)
    }).add_layer(|l| {
        MaxPooling2DBuilder::<16,28,28,2,2,0,2>::new().build(l,&device).unwrap()
    }).add_layer(|l| {
        let rnd = rnd.clone();
        ConvolutionLayerBuilder::<16,32,14,14,3,3,1,1>::new().build(l,&device, move || n2.sample(&mut rnd.borrow_mut().deref_mut())).unwrap()
    }).add_layer(|l| {
        ActivationLayer::new(l,ReLu::new(&device),&device)
    }).add_layer(|l| {
        MaxPooling2DBuilder::<32,14,14,2,2,0,2>::new().build(l,&device).unwrap()
    }).add_layer(|l| {
        BridgeLayerBuilder::<Arr<f32,{32 * 7 * 7}>>::new().build(l,&device).unwrap()
    }).add_layer(|l| {
        let rnd = rnd.clone();
        LinearLayerBuilder::new::<{32 * 7 * 7},120>().build(l,&device, move || n3.sample(&mut rnd.borrow_mut().deref_mut()), || 0.).unwrap()
    }).add_layer(|l| {
        ActivationLayer::new(l,ReLu::new(&device),&device)
    }).add_layer(|l| {
        let rnd = rnd.clone();
        LinearLayerBuilder::new::<120,10>().build(l,&device, move || n4.sample(&mut rnd.borrow_mut().deref_mut()), || 0.).unwrap()
    }).add_layer(|l| {
        ActivationLayer::new(l,SoftMax::new(&device),&device)
    }).add_layer_train(|l| {
        LinearOutputLayer::new(l,&device)
    });

    let mut teachers:Vec<(usize,PathBuf)> = Vec::new();

    for n in 0..10 {
        for entry in fs::read_dir(Path::new("mnist")
            .join("mnist_png")
            .join("training")
            .join(n.to_string())).unwrap() {
            let path = entry.unwrap().path();

            teachers.push((n,path));
        }
    }
    let mut optimizer = MomentumSGD::new(0.004);

    let mut rng = rand::thread_rng();

    teachers.shuffle(&mut rng);

    let mut correct_answers = 0;

    let mut teachers = teachers.into_iter().take(60000).collect::<Vec<(usize,PathBuf)>>();

    for _ in 0..5 {
        let mut total_loss = 0.;
        let mut count = 0;

        teachers.shuffle(&mut rng);

        for teachers in teachers.chunks(200) {
            count += 1;

            let batch_data = teachers.iter().map(|(n, path)| {
                let img = image::io::Reader::open(path).unwrap().decode().unwrap();

                let pixels = img.as_bytes().into_iter().map(|&p| p as f32 / 255.).collect::<Vec<f32>>();

                let n = *n;

                let input = Images::<f32,1,28,28>::try_from(pixels.into_boxed_slice()).unwrap();

                let mut expected = Arr::new();

                expected[n as usize] = 1.0;

                (expected, input)
            }).fold((Vec::<Arr<f32, 10>>::new(), Vec::<Images<f32,1,28,28>>::new(), ), |mut acc, (e, i)| {
                acc.0.push(e);
                acc.1.push(i.into());
                acc
            });

            let lossf = CrossEntropyMulticlass::new();

            let loss = net.batch_train(batch_data.0.into(), batch_data.1.clone().into(), &mut optimizer, &lossf).unwrap();
            total_loss += loss;

            let _ = net.batch_forward(batch_data.1.into()).unwrap();
        }
        println!("total_loss = {}", total_loss);
        println!("loss_average = {}", total_loss as f32 / count as f32);
    }

    let mut tests: Vec<(usize, PathBuf)> = Vec::new();

    for n in 0..10 {
        for entry in fs::read_dir(Path::new("mnist")
            .join("mnist_png")
            .join("testing")
            .join(n.to_string())).unwrap() {
            let path = entry.unwrap().path();

            tests.push((n, path));
        }
    }

    tests.shuffle(&mut rng);

    let count = tests.len().min(100);

    for (n, path) in tests.iter().take(100) {
        let img = image::io::Reader::open(path).unwrap().decode().unwrap();

        let pixels = img.as_bytes().into_iter().map(|&p| p as f32 / 255.).collect::<Vec<f32>>();

        let n = *n;

        let input = Images::<f32,1,28,28>::try_from(pixels.into_boxed_slice()).unwrap();

        let r = net.forward_all(input.into()).unwrap();

        let r = r.iter().enumerate().fold((0, 0.0), |acc, (n, &t)| {
            if t > acc.1 {
                (n, t)
            } else {
                acc
            }
        }).0;

        if n == r {
            correct_answers += 1;
        }
    }

    println!("correct_answers = {},{}%",correct_answers,correct_answers as f32 / count as f32 * 100.);

    debug_assert!(correct_answers as f32 / count as f32 * 100. > 80.)
}
