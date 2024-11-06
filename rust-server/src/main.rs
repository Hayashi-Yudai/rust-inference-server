use actix_web::{get, post, web, App, HttpServer, Responder};
use serde::{Deserialize, Serialize};
use tch::{no_grad, Device, IValue, Tensor};

#[derive(Serialize, Deserialize)]
struct ResponseObj {
    message: String,
    status_code: i32,
}

#[get("/ping")]
async fn ping() -> impl Responder {
    "Pong!"
}

#[post("/json")]
async fn json_post(item: web::Json<ResponseObj>) -> impl Responder {
    let output_value = predict_by_torch_model().unwrap();
    format!("{} {:?}", item.message, output_value)
}

fn predict_by_torch_model() -> Result<Vec<Vec<f64>>, Box<dyn std::error::Error>> {
    let model = tch::CModule::load_on_device("/app/src/python-model/src/model.pt", Device::Cpu)?;
    let input_tensor = Tensor::randn(&[1, 10], (tch::Kind::Float, Device::Cpu));
    let input_ivalue = IValue::Tensor(input_tensor);
    let output_tensor = no_grad(|| model.forward_is(&[input_ivalue])).unwrap();

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
    HttpServer::new(|| App::new().service(ping).service(json_post))
        .bind(("127.0.0.1", 8080))?
        .run()
        .await
}
