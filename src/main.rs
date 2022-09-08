use crate::{lpcpy::Lpc, pro::lpc_coef};
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
    const P: usize = 6; // AR次数
    let s: Vec<f32> = serde_json::from_str(include_str!("./wave.json")).unwrap();
    let mut y = s.clone();

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
    lpc.prediction_error(&s, &mut y);

    let mut rs = vec![0.0; 1440];
    lpc.inverse_prediction_error(&y, &mut rs);
    plot!("title", s, rs);
}
