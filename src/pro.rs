pub fn lpc_coef(s: &[f32], P: usize) -> Vec<f32> {
    fn auto_corr(order: usize, data: &[f32]) -> Vec<f32> {
        // 自己相関関数
        let mut acf = Vec::<f32>::new();
        for i in 0..order + 1 {
            acf.push(
                data.iter()
                    .zip(data.iter().cycle().skip(i))
                    .map(|(x, y)| x * y)
                    .sum(),
            );
        }
        acf
    }
    let r = auto_corr(P, &s);

    let mut h = vec![0.0; P + 1];
    let mut sigma = vec![0.0; P + 1];
    let mut delta = vec![0.0; P + 1];
    let mut rho = vec![0.0; P + 1];
    let mut a = vec![vec![0.0; P + 1]; P + 1];

    // レビンソン・ダービンアルゴリズム
    sigma[0] = r[0]; // sigma[0] = 時間差0の自己相関関数
    for m in 0..P {
        delta[m + 1] = r[m + 1]; // deltaの初期値 = 自己相関関数
        for i in 1..=m {
            delta[m + 1] = delta[m + 1] + a[m][i] * r[m + 1 - i]; // AR係数aを使ってdeltaを更新
        }
        if sigma[m].abs() == 0.0 || delta[m + 1].abs() > 2.0 * sigma[m].abs() {
            rho[m + 1] = 0.0; // deltaが0かsigmaの2倍より大きい場合はrho=0
            a[m + 1][m + 1] = rho[m + 1]; // AR係数をrhoにする
        } else {
            rho[m + 1] = -delta[m + 1] / sigma[m]; // 反射係数rhoの更新
            a[m + 1][m + 1] = rho[m + 1]; // AR係数a[m+1]の更新
        }
        sigma[m + 1] = sigma[m] * (1.0 - (rho[m + 1] * rho[m + 1])); // sigma[m+1]の更新
        for i in 1..=m {
            a[m + 1][i] = a[m][i] + rho[m + 1] * a[m][m + 1 - i]; // AR係数a[1]からa[m]の更新
        }
    }

    h[0] = 1.0; // 線形予測係数h[0]
    for i in 1..=P {
        h[i] = a[P][i]; // 線形予測係数h[1]からh[P]の計算
    }
    h
}
