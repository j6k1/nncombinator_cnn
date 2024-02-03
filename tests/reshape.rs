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
#[test]
fn test_im2col_1pad_2stride() {
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
        let expected = im2col::<f32, 29, 29, 3, 3, 1, 2>(&i);
        let actual = reshape::im2col::<f32, 29, 29, 3, 3, 1, 2>(&i);

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
fn test_im2col_2pad_1stride() {
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
        let expected = im2col::<f32, 29, 29, 3, 3, 2, 1>(&i);
        let actual = reshape::im2col::<f32, 29, 29, 3, 3, 2, 1>(&i);

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
fn test_im2col_0pad_1strinde() {
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
        let expected = im2col::<f32, 28, 28, 3, 3, 0, 1>(&i);
        let actual = reshape::im2col::<f32, 28, 28, 3, 3, 0, 1>(&i);

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
fn test_im2col_fh3_fw5() {
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
        let expected = im2col::<f32, 29, 29, 3, 5, 1, 1>(&i);
        let actual = reshape::im2col::<f32, 29, 29, 3, 5, 1, 1>(&i);

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
fn test_im2col_2pad_2stride_fh3_fw5() {
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
        let expected = im2col::<f32, 29, 29, 3, 5, 2, 2>(&i);
        let actual = reshape::im2col::<f32, 29, 29, 3, 5, 2, 2>(&i);

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
fn test_im2col_1pad_2stride_fh3_fw5() {
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
        let expected = im2col::<f32, 29, 29, 3, 5, 1, 2>(&i);
        let actual = reshape::im2col::<f32, 29, 29, 3, 5, 1, 2>(&i);

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
fn test_im2col_2pad_1stride_fh3_fw5() {
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
        let expected = im2col::<f32, 29, 29, 3, 5, 2, 1>(&i);
        let actual = reshape::im2col::<f32, 29, 29, 3, 5, 2, 1>(&i);

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
fn test_im2col_0pad_1strinde_fh3_fw5() {
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
        let expected = im2col::<f32, 28, 28, 3, 5, 0, 1>(&i);
        let actual = reshape::im2col::<f32, 28, 28, 3, 5, 0, 1>(&i);

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
fn test_im2col_3pad_1stride_fh3_fw5() {
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
        let expected = im2col::<f32, 29, 29, 3, 5, 3, 1>(&i);
        let actual = reshape::im2col::<f32, 29, 29, 3, 5, 3, 1>(&i);

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
fn test_im2col_5pad_1stride_fh3_fw5() {
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
        let expected = im2col::<f32, 29, 29, 3, 5, 5, 1>(&i);
        let actual = reshape::im2col::<f32, 29, 29, 3, 5, 5, 1>(&i);

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
fn test_im2col_6pad_1stride_fh3_fw5() {
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
        let expected = im2col::<f32, 29, 29, 3, 5, 6, 1>(&i);
        let actual = reshape::im2col::<f32, 29, 29, 3, 5, 6, 1>(&i);

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
fn test_im2col_10pad_1stride_fh3_fw5() {
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
        let expected = im2col::<f32, 29, 29, 3, 5, 10, 1>(&i);
        let actual = reshape::im2col::<f32, 29, 29, 3, 5, 10, 1>(&i);

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
fn test_im2col_1pad_1stride_fh3_fw5_w3_h3() {
    let mut s = Images::<f32,1,3,3>::new();

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
        let expected = im2col::<f32, 3, 3, 3, 5, 1, 1>(&i);
        let actual = reshape::im2col::<f32, 3, 3, 3, 5, 1, 1>(&i);

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
fn test_im2col_3pad_1stride_fh3_fw5_w3_h3() {
    let mut s = Images::<f32,1,3,3>::new();

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
        let expected = im2col::<f32, 3, 3, 3, 5, 3, 1>(&i);
        let actual = reshape::im2col::<f32, 3, 3, 3, 5, 3, 1>(&i);

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
fn test_im2col_4pad_1stride_fh3_fw5_w3_h3() {
    let mut s = Images::<f32,1,3,3>::new();

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
        let expected = im2col::<f32, 3, 3, 3, 5, 4, 1>(&i);
        let actual = reshape::im2col::<f32, 3, 3, 3, 5, 4, 1>(&i);

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
fn test_im2col_6pad_1stride_fh3_fw5_w3_h3() {
    let mut s = Images::<f32,1,3,3>::new();

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
        let expected = im2col::<f32, 3, 3, 3, 5, 6, 1>(&i);
        let actual = reshape::im2col::<f32, 3, 3, 3, 5, 6, 1>(&i);

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
fn test_im2col_7pad_1stride_fh3_fw5_w3_h3() {
    let mut s = Images::<f32,1,3,3>::new();

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
        let expected = im2col::<f32, 3, 3, 3, 5, 7, 1>(&i);
        let actual = reshape::im2col::<f32, 3, 3, 3, 5, 7, 1>(&i);

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
fn test_im2col_3pad_3stride_fh3_fw6_w3_h3() {
    let mut s = Images::<f32,1,3,3>::new();

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
        let expected = im2col::<f32, 3, 3, 3, 6, 3, 3>(&i);
        let actual = reshape::im2col::<f32, 3, 3, 3, 6, 3, 3>(&i);

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
fn test_im2col_5pad_2stride_fh4_fw6_w4_h4() {
    let mut s = Images::<f32,1,4,4>::new();

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
        let expected = im2col::<f32, 4, 4, 4, 6, 5, 2>(&i);
        let actual = reshape::im2col::<f32, 4, 4, 4, 6, 5, 2>(&i);

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
fn test_im2col_6pad_3stride_fh3_fw6_w3_h3() {
    let mut s = Images::<f32,1,3,3>::new();

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
        let expected = im2col::<f32, 3, 3, 3, 6, 6, 3>(&i);
        let actual = reshape::im2col::<f32, 3, 3, 3, 6, 6, 3>(&i);

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
fn test_im2col_12pad_3stride_fh4_fw10_w4_h4() {
    let mut s = Images::<f32,1,4,4>::new();

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
        let expected = im2col::<f32, 4, 4, 4, 10, 12, 3>(&i);
        let actual = reshape::im2col::<f32, 4, 4, 4, 10, 12, 3>(&i);

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
fn test_im2col_2pad_2stride_fh5_fw3() {
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
        let expected = im2col::<f32, 29, 29, 5, 3, 2, 2>(&i);
        let actual = reshape::im2col::<f32, 29, 29, 5, 3, 2, 2>(&i);

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
fn test_im2col_1pad_2stride_fh5_fw3() {
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
        let expected = im2col::<f32, 29, 29, 5, 3, 1, 2>(&i);
        let actual = reshape::im2col::<f32, 29, 29, 5, 3, 1, 2>(&i);

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
fn test_im2col_2pad_1stride_fh5_fw3() {
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
        let expected = im2col::<f32, 29, 29, 5, 3, 2, 1>(&i);
        let actual = reshape::im2col::<f32, 29, 29, 5, 3, 2, 1>(&i);

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
fn test_im2col_0pad_1strinde_fh5_fw3() {
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
        let expected = im2col::<f32, 28, 28, 5, 3, 0, 1>(&i);
        let actual = reshape::im2col::<f32, 28, 28, 5, 3, 0, 1>(&i);

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
fn test_im2col_3pad_1stride_fh5_fw3() {
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
        let expected = im2col::<f32, 29, 29, 5, 3, 3, 1>(&i);
        let actual = reshape::im2col::<f32, 29, 29, 5, 3, 3, 1>(&i);

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
fn test_im2col_5pad_1stride_fh5_fw3() {
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
        let expected = im2col::<f32, 29, 29, 5, 3, 5, 1>(&i);
        let actual = reshape::im2col::<f32, 29, 29, 5, 3, 5, 1>(&i);

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
fn test_im2col_6pad_1stride_fh5_fw3() {
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
        let expected = im2col::<f32, 29, 29, 5, 3, 6, 1>(&i);
        let actual = reshape::im2col::<f32, 29, 29, 5, 3, 6, 1>(&i);

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
fn test_im2col_10pad_1stride_fh5_fw3() {
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
        let expected = im2col::<f32, 29, 29, 5, 3, 10, 1>(&i);
        let actual = reshape::im2col::<f32, 29, 29, 5, 3, 10, 1>(&i);

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
fn test_im2col_1pad_1stride_fh5_fw3_w3_h3() {
    let mut s = Images::<f32,1,3,3>::new();

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
        let expected = im2col::<f32, 3, 3, 5, 3, 1, 1>(&i);
        let actual = reshape::im2col::<f32, 3, 3, 5, 3, 1, 1>(&i);

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
fn test_im2col_3pad_1stride_fh5_fw3_w3_h3() {
    let mut s = Images::<f32,1,3,3>::new();

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
        let expected = im2col::<f32, 3, 3, 5, 3, 3, 1>(&i);
        let actual = reshape::im2col::<f32, 3, 3, 5, 3, 3, 1>(&i);

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
fn test_im2col_4pad_1stride_fh5_fw3_w3_h3() {
    let mut s = Images::<f32,1,3,3>::new();

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
        let expected = im2col::<f32, 3, 3, 5, 3, 4, 1>(&i);
        let actual = reshape::im2col::<f32, 3, 3, 5, 3, 4, 1>(&i);

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
fn test_im2col_6pad_1stride_fh5_fw3_w3_h3() {
    let mut s = Images::<f32,1,3,3>::new();

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
        let expected = im2col::<f32, 3, 3, 5, 3, 6, 1>(&i);
        let actual = reshape::im2col::<f32, 3, 3, 5, 3, 6, 1>(&i);

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
fn test_im2col_7pad_1stride_fh5_fw3_w3_h3() {
    let mut s = Images::<f32,1,3,3>::new();

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
        let expected = im2col::<f32, 3, 3, 5, 3, 7, 1>(&i);
        let actual = reshape::im2col::<f32, 3, 3, 5, 3, 7, 1>(&i);

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
fn test_im2col_3pad_3stride_fh6_fw3_w3_h3() {
    let mut s = Images::<f32,1,3,3>::new();

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
        let expected = im2col::<f32, 3, 3, 6, 3, 3, 3>(&i);
        let actual = reshape::im2col::<f32, 3, 3, 6, 3, 3, 3>(&i);

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
fn test_im2col_5pad_2stride_fh6_fw4_w4_h4() {
    let mut s = Images::<f32,1,4,4>::new();

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
        let expected = im2col::<f32, 4, 4, 6, 4, 5, 2>(&i);
        let actual = reshape::im2col::<f32, 4, 4, 6, 4, 5, 2>(&i);

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
fn test_im2col_6pad_3stride_fh6_fw3_w3_h3() {
    let mut s = Images::<f32,1,3,3>::new();

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
        let expected = im2col::<f32, 3, 3, 6, 3, 6, 3>(&i);
        let actual = reshape::im2col::<f32, 3, 3, 6, 3, 6, 3>(&i);

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
fn test_im2col_12pad_3stride_fh10_fw4_w4_h4() {
    let mut s = Images::<f32,1,4,4>::new();

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
        let expected = im2col::<f32, 4, 4, 10, 4, 12, 3>(&i);
        let actual = reshape::im2col::<f32, 4, 4, 10, 4, 12, 3>(&i);

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
fn test_im2col_3pad_1stride_fh3_fw3() {
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
        let expected = im2col::<f32, 29, 29, 3, 3, 3, 1>(&i);
        let actual = reshape::im2col::<f32, 29, 29, 3, 3, 3, 1>(&i);

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
fn test_im2col_5pad_1stride_fh3_fw3() {
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
        let expected = im2col::<f32, 29, 29, 3, 3, 5, 1>(&i);
        let actual = reshape::im2col::<f32, 29, 29, 3, 3, 5, 1>(&i);

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
fn test_im2col_6pad_1stride_fh3_fw3() {
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
        let expected = im2col::<f32, 29, 29, 3, 3, 6, 1>(&i);
        let actual = reshape::im2col::<f32, 29, 29, 3, 3, 6, 1>(&i);

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
fn test_im2col_10pad_1stride_fh3_fw3() {
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
        let expected = im2col::<f32, 29, 29, 3, 3, 10, 1>(&i);
        let actual = reshape::im2col::<f32, 29, 29, 3, 3, 10, 1>(&i);

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
fn test_im2col_1pad_1stride_fh3_fw3_w3_h3() {
    let mut s = Images::<f32,1,3,3>::new();

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
        let expected = im2col::<f32, 3, 3, 3, 3, 1, 1>(&i);
        let actual = reshape::im2col::<f32, 3, 3, 3, 3, 1, 1>(&i);

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
fn test_im2col_3pad_1stride_fh3_fw3_w3_h3() {
    let mut s = Images::<f32,1,3,3>::new();

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
        let expected = im2col::<f32, 3, 3, 3, 3, 3, 1>(&i);
        let actual = reshape::im2col::<f32, 3, 3, 3, 3, 3, 1>(&i);

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
fn test_im2col_4pad_1stride_fh3_fw3_w3_h3() {
    let mut s = Images::<f32,1,3,3>::new();

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
        let expected = im2col::<f32, 3, 3, 3, 3, 4, 1>(&i);
        let actual = reshape::im2col::<f32, 3, 3, 3, 3, 4, 1>(&i);

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
fn test_im2col_6pad_1stride_fh3_fw3_w3_h3() {
    let mut s = Images::<f32,1,3,3>::new();

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
        let expected = im2col::<f32, 3, 3, 3, 3, 6, 1>(&i);
        let actual = reshape::im2col::<f32, 3, 3, 3, 3, 6, 1>(&i);

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
fn test_im2col_7pad_1stride_fh3_fw3_w3_h3() {
    let mut s = Images::<f32,1,3,3>::new();

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
        let expected = im2col::<f32, 3, 3, 3, 3, 7, 1>(&i);
        let actual = reshape::im2col::<f32, 3, 3, 3, 3, 7, 1>(&i);

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
fn test_im2col_3pad_3stride_fh3_fw3_w3_h3() {
    let mut s = Images::<f32,1,3,3>::new();

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
        let expected = im2col::<f32, 3, 3, 3, 3, 3, 3>(&i);
        let actual = reshape::im2col::<f32, 3, 3, 3, 3, 3, 3>(&i);

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
fn test_im2col_5pad_2stride_fh4_fw4_w4_h4() {
    let mut s = Images::<f32,1,4,4>::new();

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
        let expected = im2col::<f32, 4, 4, 4, 4, 5, 2>(&i);
        let actual = reshape::im2col::<f32, 4, 4, 4, 4, 5, 2>(&i);

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
fn test_im2col_6pad_3stride_fh3_fw3_w3_h3() {
    let mut s = Images::<f32,1,3,3>::new();

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
        let expected = im2col::<f32, 3, 3, 3, 3, 6, 3>(&i);
        let actual = reshape::im2col::<f32, 3, 3, 3, 3, 6, 3>(&i);

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
fn test_im2col_12pad_3stride_fh4_fw4_w4_h4() {
    let mut s = Images::<f32,1,4,4>::new();

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
        let expected = im2col::<f32, 4, 4, 4, 4, 12, 3>(&i);
        let actual = reshape::im2col::<f32, 4, 4, 4, 4, 12, 3>(&i);

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