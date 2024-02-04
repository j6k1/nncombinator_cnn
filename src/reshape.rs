use itertools::Itertools;
use nncombinator::arr::Arr2;
use nncombinator::mem::AsRawSlice;
use crate::{Assert, assert_convolution, IsTrue};

use crate::collection::{ImageView};

#[inline]
pub fn im2col<'a,T,const H:usize,const W:usize,const FH:usize,const FW:usize,const PAD:usize,const S:usize>(image:&ImageView<'a,T,H,W>)
    -> Arr2<T,{ ((H + PAD * 2 - FH) / S + 1) * ((W + PAD * 2 - FW) / S + 1) }, { FH * FW }>
    where T: Default + Clone + Copy + Send + Sync + 'static,
          Assert<{ assert_convolution::<H,W,FH,FW,PAD,S>() }>: IsTrue {
    let xs = (W + PAD * 2 - FW) / S + 1;

    let mut r = Arr2::<T,{ ((H + PAD * 2 - FH) / S + 1) * ((W + PAD * 2 - FW) / S + 1) }, { FH * FW }>::new();

    for y in 0..((H + PAD * 2 - FH) / S + 1) {
        for x in 0..((W + PAD * 2 - FW) / S + 1) {
            if FW * FH >= 8 {
                let chunk_size = FW * FH / 8;

                for c in 0..8 {
                    for index in 0..chunk_size {
                        let index = c * chunk_size + index;

                        let i = index / FW;
                        let j = index - i * FW;
                        let sy = y * S + i;
                        let sx = x * S + j;

                        if sy >= PAD && sy < H + PAD && sx >= PAD && sx < W + PAD {
                            r[(y * xs + x, i * FW + j)] = image[(
                                sy - PAD, sx - PAD
                            )];
                        }
                    }
                }

                for index in (chunk_size * 8)..(FH * FW) {
                    let i = index / FW;
                    let j = index - i * FW;
                    let sy = y * S + i;
                    let sx = x * S + j;

                    if sy >= PAD && sy < H + PAD && sx >= PAD && sx < W + PAD {
                        r[(y * xs + x, i * FW + j)] = image[(
                            sy - PAD, sx - PAD
                        )];
                    }
                }
            } else {
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
    }

    r
}
