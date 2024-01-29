#![allow(unused_attributes)]
#![allow(incomplete_features)]
#![feature(generic_const_exprs)]

use nncombinator::arr::Arr2;
use nncombinator_cnn::collection::{Images, ImageView};
use nncombinator_cnn::reshape;

#[test]
fn test_im2col() {
    let mut s = Images::<f32,1,28,28>::new();

    let mut c = 1;

    for mut i in s.iter_mut() {
        for mut i in i.iter_mut() {
            for i in i.iter_mut() {
                *i = c as f32;
                c += 1;
            }
        }
    }

    for i in s.iter() {
        let expected = im2col::<f32, 28, 28, 3, 3, 1, 1>(&i);
        let actual = reshape::im2col::<f32, 28, 28, 3, 3, 1, 1>(&i);

        for (i,(e,a)) in expected.iter().zip(actual.iter()).enumerate() {
            for (j,(e,a)) in e.iter().zip(a.iter()).enumerate() {
                if e != a {
                    dbg!(i,j);
                }
                assert_eq!(e,a);
            }
        }
    }
}

#[test]
fn test_im2col_2pad_2stride() {
    let mut s = Images::<f32,1,29,29>::new();

    let mut c = 1;

    for mut i in s.iter_mut() {
        for mut i in i.iter_mut() {
            for i in i.iter_mut() {
                *i = c as f32;
                c += 1;
            }
        }
    }

    for i in s.iter() {
        let expected = im2col::<f32, 29, 29, 3, 3, 2, 2>(&i);
        let actual = reshape::im2col::<f32, 29, 29, 3, 3, 2, 2>(&i);

        for (i,(e,a)) in expected.iter().zip(actual.iter()).enumerate() {
            for (j,(e,a)) in e.iter().zip(a.iter()).enumerate() {
                if e != a {
                    dbg!(i,j);
                }
                assert_eq!(e,a);
            }
        }
    }
}
fn im2col<'a,T,const H:usize,const W:usize,const FH:usize,const FW:usize,const PAD:usize,const S:usize>(image:&ImageView<'a,T,H,W>)
    -> Arr2<T,{ ((H + PAD * 2 - FH) / S + 1) * ((W + PAD * 2 - FW) / S + 1) }, { FH * FW }>
    where T: Default + Clone + Copy + Send + Sync + 'static {
    let xs = (W + PAD * 2 - FW) / S + 1;

    let mut r = Arr2::<T,{ ((H + PAD * 2 - FH) / S + 1) * ((W + PAD * 2 - FW) / S + 1) }, { FH * FW }>::new();

    for y in 0..((H + PAD * 2 - FH) / S + 1) {
        for x in 0..((W + PAD * 2 - FW) / S + 1) {
            for i in 0..FH {
                for j in 0..FW {
                    let sy = y * S + i;
                    let sx = x * S + j;

                    if sy >= PAD && sy < H + PAD && sx >= PAD && sx < W + PAD {
                        r[(y * xs + x, i * FW + j)] = image[(
                            sy - PAD, sx - PAD
                        )];
                    }
                }
            }
        }
    }

    r
}