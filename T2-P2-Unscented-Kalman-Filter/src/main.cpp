//main.cpp
#include <uWS/uWS.h>
#include <iostream>
#include "json.hpp"
#include <math.h>
#include "ukf.h"
#include "tools.h"

using namespace std;

// for convenience
using json = nlohmann::json;

// Checks if the SocketIO event has JSON data.
// If there is data the JSON object in string format will be returned,
// else the empty string "" will be returned.
std::string hasData(std::string s) {
  auto found_null = s.find("null");
  auto b1 = s.find_first_of("[");
  auto b2 = s.find_first_of("]");
  if (found_null != std::string::npos) {
    return "";
  }
  else if (b1 != std::string::npos && b2 != std::string::npos) {
    return s.substr(b1, b2 - b1 + 1);
  }
  return "";
}
extern void gen_measurement_gt_values (VectorXd &gt_values, istringstream &iss, MeasurementPackage &mess_package);
extern void gen_estimate (VectorXd &estimte, UKF &ukf);


int	gLidarRadarFlag = 3;
double 	g_std_a = 1.0;
double	g_std_ywdd = 0.5;
string 	nis_output_file = "/tmp/NIS_output";
ofstream out_file_;

void check_arguments(int argc, char* argv[]) {
  bool valid_args = false;

  // make sure the user has provided input and output files
  if (argc == 1) {
        cout << "ukf <radar_lidar=1|2|3> <std_a> <stdyawdd> <NIS_output_file>" <<endl;
	gLidarRadarFlag = 3;
	g_std_a    = 1.0;
	g_std_ywdd = 0.5;
	nis_output_file = "/tmp/NIS_output";
	valid_args = true;
	cout << "ukf <radar_lidar= " << gLidarRadarFlag << ">";
        cout << "<std_a= " << g_std_a << "> ";
        cout << "<std_yawdd= " << g_std_ywdd << "> ";
        cout << "<NIS_output_file= " << nis_output_file << ">" <<endl;
  } else if (argc == 5) {
	gLidarRadarFlag = stoi(argv[1]);
        std::istringstream iss2(argv[2]);
	iss2 >> g_std_a    ; 
        std::istringstream iss3( argv[3]);
	iss3 >> g_std_ywdd ;
	nis_output_file = argv[4];
        valid_args      = true;
	cout << "ukf <radar_lidar= " << gLidarRadarFlag << ">";
        cout << "<std_a= " << g_std_a << "> ";
        cout << "<std_yawdd= " << g_std_ywdd << "> ";
        cout << "<NIS_output_file= " << nis_output_file << ">" <<endl;
  } else {
    cout << "ukf <radar_lidar=1|2|3> <std_a=0.5> <std_yawdd=1> <NIS_output_file=/tmp/NIS_output>" << endl;
  }

  if (!valid_args) {
    exit(EXIT_FAILURE);
  }
}

int main(int argc, char* argv[])
{
  uWS::Hub h;

   check_arguments(argc, argv);
  // Create a Kalman Filter instance
  UKF ukf;
  
  ukf.tune_parameters (gLidarRadarFlag, g_std_a, g_std_ywdd);
 
  //out_file_.open(nis_output_file);
  // used to compute the RMSE later
  Tools tools;
  vector<VectorXd> estimations;
  vector<VectorXd> ground_truth;

  h.onMessage([&ukf,&tools,&estimations,&ground_truth](uWS::WebSocket<uWS::SERVER> ws, char *data, size_t length, uWS::OpCode opCode) {
    // "42" at the start of the message means there's a websocket message event.
    // The 4 signifies a websocket message
    // The 2 signifies a websocket event

    if (length && length > 2 && data[0] == '4' && data[1] == '2')
    {

      auto s = hasData(std::string(data));
      if (s != "") {
      	
        auto j = json::parse(s);

        std::string event = j[0].get<std::string>();
        
        if (event == "telemetry") {
          // j[1] is the data JSON object
          
          string sensor_measurment = j[1]["sensor_measurement"];
          
          MeasurementPackage meas_package;
          istringstream iss(sensor_measurment);

    	  VectorXd gt_values(4);
          gen_measurement_gt_values (gt_values, iss, meas_package);

    	  ground_truth.push_back(gt_values);
          
          //Call ProcessMeasurment(meas_package) for Kalman filter
    	  ukf.ProcessMeasurement(meas_package);    	  

    	  //Push the current estimated x,y positon from the Kalman filter's state vector

    	  VectorXd estimate(4);

          out_file_.open(nis_output_file, std::ios_base::app );
    	  if (meas_package.sensor_type_ == MeasurementPackage::LASER) {
              out_file_ << "L " << ukf.NIS_lidar_ << "\n";
          } else if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {
              out_file_ << "R " << ukf.NIS_radar_ << "\n";
          }
          out_file_.close();
  
          gen_estimate (estimate, ukf);
    	  estimations.push_back(estimate);
          //cout << "ESTIMATE \n" << estimate << endl;
    	  VectorXd RMSE = tools.CalculateRMSE(estimations, ground_truth);
          cout << "RMSE : \n" << RMSE << "\ndone RMSE\n";
          json msgJson;
    	  double p_x = ukf.x_(0);
    	  double p_y = ukf.x_(1);
          msgJson["estimate_x"] = p_x;
          msgJson["estimate_y"] = p_y;
          msgJson["rmse_x"] =  RMSE(0);
          msgJson["rmse_y"] =  RMSE(1);
          msgJson["rmse_vx"] = RMSE(2);
          msgJson["rmse_vy"] = RMSE(3);
          auto msg = "42[\"estimate_marker\"," + msgJson.dump() + "]";
          // std::cout << msg << std::endl;
          ws.send(msg.data(), msg.length(), uWS::OpCode::TEXT);
          
	  
        } //event == telemetry
      } else {
        
        std::string msg = "42[\"manual\",{}]";
        ws.send(msg.data(), msg.length(), uWS::OpCode::TEXT);
      }
    } // Valid Data

  }); // On receiving Message from Simulator

  // We don't need this since we're not using HTTP but if it's removed the program
  // doesn't compile :-(
  h.onHttpRequest([](uWS::HttpResponse *res, uWS::HttpRequest req, char *data, size_t, size_t) {
    const std::string s = "<h1>Hello world!</h1>";
    if (req.getUrl().valueLength == 1)
    {
      res->end(s.data(), s.length());
    }
    else
    {
      // i guess this should be done more gracefully?
      res->end(nullptr, 0);
    }
  });

  h.onConnection([&h](uWS::WebSocket<uWS::SERVER> ws, uWS::HttpRequest req) {
    std::cout << "Connected!!!" << std::endl;
  });

  h.onDisconnection([&h](uWS::WebSocket<uWS::SERVER> ws, int code, char *message, size_t length) {
    out_file_.close();
    ws.close();
    std::cout << "Disconnected" << std::endl;
  });

  int port = 4567;
  if (h.listen(port))
  {
    std::cout << "Listening to port " << port << std::endl;
  }
  else
  {
    std::cerr << "Failed to listen to port" << std::endl;
    return -1;
  }
  h.run();
}

///////////////////////////////////////////////////////////////////////////////
// Read input from simulator and initialize measurement package
//////////////////////////////////////////////////////////////////////////////
void gen_measurement_gt_values (VectorXd &gt_values, istringstream &iss, MeasurementPackage &meas_package)
{

    	  // reads first element from the current line
    	  long long timestamp;
    	  string sensor_type;
    	  iss >> sensor_type;

    	  if (sensor_type.compare("L") == 0) {
      	  		meas_package.sensor_type_ = MeasurementPackage::LASER;
          		meas_package.raw_measurements_ = VectorXd(2);
          		float px;
      	  		float py;
          		iss >> px;
          		iss >> py;
          		meas_package.raw_measurements_ << px, py;
          		iss >> timestamp;
          		meas_package.timestamp_ = timestamp;
          } else if (sensor_type.compare("R") == 0) {

      	  		meas_package.sensor_type_ = MeasurementPackage::RADAR;
          		meas_package.raw_measurements_ = VectorXd(3);
          		float ro;
      	  		float theta;
      	  		float ro_dot;
          		iss >> ro;
          		iss >> theta;
          		iss >> ro_dot;
          		meas_package.raw_measurements_ << ro,theta, ro_dot;
          		iss >> timestamp;
          		meas_package.timestamp_ = timestamp;
          }
          float x_gt;
    	  float y_gt;
    	  float vx_gt;
    	  float vy_gt;
    	  iss >> x_gt;
    	  iss >> y_gt;
    	  iss >> vx_gt;
    	  iss >> vy_gt;

    	  //VectorXd gt_values(4);
    	  gt_values(0) = x_gt;
    	  gt_values(1) = y_gt; 
    	  gt_values(2) = vx_gt;
    	  gt_values(3) = vy_gt;
}

///////////////////////////////////////////////////////////////////////////////
// After UKF Process management computation is done.
// generate estimate value and send back to simulator                 
//////////////////////////////////////////////////////////////////////////////
void gen_estimate (VectorXd &estimate, UKF &ukf)
{

    	  double p_x = ukf.x_(0);
    	  double p_y = ukf.x_(1);
    	  double v   = ukf.x_(2);
    	  double yaw = ukf.x_(3);

    	  double v1 = cos(yaw)*v;
    	  double v2 = sin(yaw)*v;

    	  estimate(0) = p_x;
    	  estimate(1) = p_y;
    	  estimate(2) = v1;
    	  estimate(3) = v2;
}

