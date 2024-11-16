use actix_web::{get, post, web, App, HttpServer, Responder};
use serde::{Deserialize, Serialize};
use tch::{no_grad, Device, IValue, Tensor, CModule};
use std::sync::LazyLock;

#[derive(Serialize, Deserialize)]
struct ResponseObj {
    message: String,
    probability: f64,
}

#[derive(Serialize, Deserialize)]
struct TitanicInputData {
    pclass: i64,
    sex: String,
    age: i64,
    sibsp: i64,
    parch: i64,
    embarked: String,
}

static MODEL: LazyLock<CModule> = LazyLock::new(|| {
    let device = Device::cuda_if_available();
    // println!("Device: {:?}", device);
    CModule::load_on_device("/app/src/python-model/model.pt", device).unwrap()
});

#[get("/ping")]
async fn ping() -> impl Responder {
    "Pong!"
}

#[post("/predict")]
async fn predict_titanic_survival(item: web::Json<TitanicInputData>) -> impl Responder {
    let input_value = preprocess_input_date(item.into_inner());
    let output_value = predict_by_torch_model(input_value).unwrap();

    if output_value[0][0] >= 0.5 {
        return web::Json(ResponseObj {
            message: "Died".to_string(),
            probability: output_value[0][0],
        });
    } else {
        return web::Json(ResponseObj {
            message: "Survived".to_string(),
            probability: output_value[0][1],
        });
    }
}

fn preprocess_input_date(item: TitanicInputData) -> Vec<Vec<f64>> {
    let mut input_data: Vec<f64> = vec![0.0; 10];
    // normalize the numerical values
    input_data[0] = item.pclass as f64 / 3.0;
    input_data[1] = item.age as f64 / 100.0;
    input_data[2] = item.sibsp as f64 / 8.0;
    input_data[3] = item.parch as f64 / 6.0;

    // one-hot encode the categorical values
    if item.sex == "male" {
        input_data[4] = 1.0;
    } else {
        input_data[5] = 1.0;
    }

    if item.embarked == "C" {
        input_data[6] = 1.0;
    } else if item.embarked == "Q" {
        input_data[7] = 1.0;
    } else if item.embarked == "S" {
        input_data[8] = 1.0;
    } else {
        input_data[9] = 1.0;
    }

    vec![input_data]
}

fn predict_by_torch_model(input: Vec<Vec<f64>>) -> Result<Vec<Vec<f64>>, Box<dyn std::error::Error>> {
    let flatten_input: Vec<f64> = input.into_iter().flatten().collect();
    let input_tensor = Tensor::from_slice(&flatten_input).view([1, 10]).to_kind(tch::Kind::Float);

    let input_ivalue = IValue::Tensor(input_tensor);
    let output_tensor = no_grad(|| MODEL.forward_is(&[input_ivalue])).unwrap();

    match output_tensor {
        IValue::Tensor(t) => {
            let values: Vec<Vec<f64>> = Vec::try_from(&t)?;
            Ok(values)
        },
        _ => {
            Err("Output is not a tensor".into())
        }
    }
}

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    HttpServer::new(|| App::new().service(ping).service(predict_titanic_survival))
        .bind(("0.0.0.0", 8080))?
        .run()
        .await
}