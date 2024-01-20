use rand::{prelude, Rng, SeedableRng};
use rand_xorshift::XorShiftRng;
use nncombinator_cnn::collection::{Images, ImageView};
use nncombinator_cnn::reshape;

#[test]
fn test_im2col() {
    let mut rnd = prelude::thread_rng();

    let mut rnd = XorShiftRng::from_seed(rnd.gen());

    let mut s = Images::<f32,1,28,28>::new();

    for mut i in s.iter_mut() {
        for mut i in i.iter_mut() {
            for i in i.iter_mut() {
                *i = rnd.gen();
            }
        }
    }

    for i in s.iter() {
        let expected = im2col::<f32, 28, 28, 3, 3, 1, 1>(&i);
        let actual = reshape::im2col::<f32, 28, 28, 3, 3, 1, 1>(&i);

        for (i,(e,a)) in expected.iter().zip(actual.iter()).enumerate() {
            if e != a {
                dbg!(i);
            }
            assert_eq!(e,a);
        }
    }
}
fn im2col<'a,T,const H:usize,const W:usize,const FH:usize,const FW:usize,const PAD:usize,const S:usize>(image:&ImageView<'a,T,H,W>)
    -> Vec<T> where T: Default + Clone + Copy + Send + Sync + 'static {
    let ys = ((H + PAD * 2 - FH) / S + 1) * FH;
    let xs = ((W + PAD * 2 - FW) / S + 1) * FW;

    let mut r = vec![T::default(); ys * xs];

    for y in (0..(H - FH + PAD * 2)).step_by(S) {
        for x in (0..(W - FW + PAD * 2)).step_by(S) {
            for i in y..(y + FH) {
                for j in x..(x + FW) {
                    if y + i >= PAD && y + i < H - FH + PAD && x + j >= PAD && x + j < W - FW + PAD {
                        r[(y * FH / S + i) * xs + x * FW / S + j] = image[(y + i, x + j)];
                    }
                }
            }
        }
    }

    r
}