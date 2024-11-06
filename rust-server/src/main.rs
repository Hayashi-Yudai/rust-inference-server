// use actix_web::{get, post, web, App, HttpServer, Responder};
// use serde::{Deserialize, Serialize};
// 
// #[derive(Serialize, Deserialize)]
// struct ResponseObj {
//     message: String,
//     status_code: i32,
// }
// 
// #[get("/hello/{name}")]
// async fn greet(name: web::Path<String>) -> impl Responder {
//     format!("Hello {name}!")
// }
// 
// #[get("/json")]
// async fn json() -> impl Responder {
//     web::Json(ResponseObj {
//         message: "Hello, World!".to_string(),
//         status_code: 200,
//     })
// }
// 
// #[post("/json")]
// async fn json_post(item: web::Json<ResponseObj>) -> impl Responder {
//     format!("{} {}", item.message, item.status_code)
// }
// 
// #[actix_web::main]
// async fn main() -> std::io::Result<()> {
//     HttpServer::new(|| App::new().service(greet).service(json).service(json_post))
//         .bind(("127.0.0.1", 8080))?
//         .run()
//         .await
// }

use tch::Device;
use tch::TrainableCModule;
use tch::nn::VarStore;

fn main() {
    let vs = VarStore::new(Device::Cpu);
    let model = TrainableCModule::load("../python-model/src/model.pt", vs.root());
}
