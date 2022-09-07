use crate::{lpcpy::Lpc, pro::lpc_coef};
use realfft::num_complex::ComplexFloat;
mod lpcpy;
mod pro;
macro_rules! plot_with_name {
    ($title:expr, $($name_ys:expr), +) => {
        use plotly::common::{Mode, Title};
        use plotly::layout::Layout;
        use plotly::{Plot, Scatter};
        let layout = Layout::new().title(Title::new($title));
        let mut plot = Plot::new();
        $(
            let (name, ys) = $name_ys;
            let xs = (0..ys.len()).map(|i| i as f32);
            let trace = Scatter::new(xs, ys.to_owned())
                .mode(Mode::Lines)
                .name(name);
            plot.add_trace(trace);
        )+
        plot.set_layout(layout);
        plot.show();
    };
}
macro_rules! plot {
    ($title:expr, $($ys:expr), +) => {
        plot_with_name!($title, $((stringify!($ys), $ys)   ), +);
    }
}
fn main() {
    const P: usize = 28; // AR次数
    let s: Vec<f32> = serde_json::from_str(include_str!("./wave.json")).unwrap();
    let N = s.len();
    let mut y = s.clone();

    let t = 0;
    // for t in 0..N {
    //************************************************************************//

    ////////////////////////////////////////////////////
    //                                                //
    //              Signal Processing                 //
    //                                                //
    //  現在時刻tの入力 s[t] から出力 y[t] をつくる   //
    //                                                //
    //  ※ tは0からMEM_SIZE-1までをループ             //
    //                                                //
    ////////////////////////////////////////////////////
    let mut lpc = Lpc::new(P);
    lpc.calc(&s);
    println!("{:?}", lpc.coef());
    let h = lpc.coef();
    for (t, y) in y.iter_mut().enumerate() {
        let mut e = 0.0;
        for i in (1..=P).rev() {
            e = e + h[i] * s[(1440 + t - i) % 1440]; // 導出した線形予測係数から予測誤差を計算
        }
        *y = e; // 予測誤差(音源)を出力とする
    }
    let mut y2 = y.clone();
    lpc.prediction_error(&s, &mut y2);
    plot!("title", y, y2);
}
