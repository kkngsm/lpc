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
    const LPC_ORDER: usize = 16;

    let original = (0..128)
        .map(|i| {
            let i = i as f32;
            (i * 0.01).sin()
                + 0.75 * (i * 0.03).sin()
                + 0.5 * (i * 0.05).sin()
                + 0.25 * (i * 0.11).sin()
        })
        .collect::<Vec<_>>();
    // LPCで前向き予測した信号を求める
    let mut predicted = original.clone();
    // 過去lpcOrder分から予測するので開始インデックスはlpcOrderから
    // それより前は予測せずにオリジナルの信号をコピーしている

    let r = auto_corr(LPC_ORDER, &original);
    let (a, _e) = levinson_durbin(LPC_ORDER, &r);
    for (i, p) in predicted.iter_mut().enumerate().skip(LPC_ORDER) {
        *p = -original
            .iter()
            .skip(i)
            .take(LPC_ORDER)
            .zip(a.iter().skip(1))
            .map(|(x, y)| x * y)
            .sum::<f32>();
    }
    plot!("title", original, predicted);
}
fn auto_corr(order: usize, data: &[f32]) -> Vec<f32> {
    // 自己相関関数
    let lags_num = order + 1;
    let mut r: Vec<f32> = Vec::with_capacity(lags_num);
    for i in 0..lags_num {
        r.push(data.iter().zip(&data[i..]).map(|(x, y)| x * y).sum());
    }
    r
}
fn levinson_durbin(order: usize, r: &[f32]) -> (Vec<f32>, Vec<f32>) {
    // Levinson-Durbinのアルゴリズム
    // k次のLPC係数からk+1次のLPC係数を再帰的に計算して

    // LPC係数を求める
    // LPC係数（再帰的に更新される）
    // a[0]は1で固定のためlpcOrder個の係数を得るためには+1が必要
    // let mut a = vec![0.0; order + 1];
    // a[0] = 1.0;
    // a[1] = -r[1] / r[0];

    let mut a = Vec::with_capacity(order + 1);
    a.push(1.0);
    a.push(-r[1] / r[0]);

    // 最小誤差
    let mut e = Vec::with_capacity(order + 1);
    e.push(1.0);
    e.push(r[0] + r[1] * a[1]);

    let mut U = Vec::with_capacity(order + 2);
    let mut V = Vec::with_capacity(order + 2);

    // kの場合からk=1の場合までを再帰的に求める
    for k in 1..order {
        //lamdaを更新
        let mut lambda = 0.0;
        for j in 0..=k {
            lambda -= a[j] * r[k + 1 - j];
        }
        lambda /= e[k];
        // aを更新
        // UとVからaを更新
        U.push(1.0);
        V.push(0.0);
        for x in &a[1..=k] {
            U.push(*x);
        }
        for x in a.iter().skip(1).take(k).rev() {
            V.push(*x);
        }
        U.push(0.0);
        V.push(1.0);

        a.clear();
        for (u, v) in U.iter().zip(&V) {
            a.push(u + lambda * v)
        }

        e.push(e[k] * (1.0 - lambda * lambda));
        U.clear();
        V.clear();
    }
    (a, e)
}
