//CarND-Kidnapped-Vehicle-Project/particle_filter.cpp
/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;
#define EPSILON 0.001
#define NUM_PARTICLE 25 
#include "my_module"

////////////////////////////////////////////////////
    // TODO: 
    // o Set the number of particles. 
    // Initialize all particles to first position (based on estimates of x, y, theta and 
    // set their uncertainties from GPS) and 
    // set all weights to 1. 
    // Add random Gaussian noise to each particle.
    // NOTE: Consult particle_filter.h for more information about this method (and others in this file).
    // x,y, theta is from GPS
void ParticleFilter::init(double x, double y, double theta, double std[], double std_landmark[]) {

    num_particles  = NUM_PARTICLE;
    is_initialized = true;
    packet_count   = 0;

    default_random_engine gen;

    for (int i = 0; i < num_particles; ++i) {
        // Add generated particle data to particles class
        Particle my_particle;
        my_particle.id     = i;
        addRandomNoise (my_particle, x, std[0], y, std[1], theta, std[2]);
        my_particle.weight = 1.0;
        particles.push_back(my_particle);
	weights.push_back  (my_particle.weight);
    }
    gauss_norm       = (2 * M_PI * std_landmark[0] * std_landmark[1]);
    two_sig_x_square = (2 * pow(std_landmark[0], 2));
    two_sig_y_square = (2 * pow(std_landmark[1], 2));

    //cout << "Particle : " << particles [10].x << " " << particles[10].theta << endl;
}

// TODO: 
//    Add measurements to each particle and 
//    add random Gaussian noise.
// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
//  http://www.cplusplus.com/reference/random/default_random_engine/
void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
    for (int i = 0; i < num_particles; i++) {
        double p_x     = particles[i].x; double p_y     = particles[i].y; double p_theta = particles[i].theta;
	double pred_x; double pred_y; double pred_theta;

        if (fabs(yaw_rate) < EPSILON) {
            pred_x     = p_x + velocity * cos(p_theta) * delta_t;
	    pred_y     = p_y + velocity * sin(p_theta) * delta_t;
	    pred_theta = p_theta;
	} else {
	    pred_x     = p_x + (velocity/yaw_rate) * (sin(p_theta + (yaw_rate * delta_t)) - sin(p_theta));
	    pred_y     = p_y + (velocity/yaw_rate) * (cos(p_theta) - cos(p_theta + (yaw_rate * delta_t)));
	    pred_theta = p_theta + (yaw_rate * delta_t);
	}
        addRandomNoise (particles[i], pred_x, std_pos[0], pred_y, std_pos[1], pred_theta, std_pos[2]);
    }
}

// TODO: 
//     Find the predicted measurement that is closest to each observed measurement and 
//     assign the observed measurement to this particular landmark.
// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
//   implement this method and use it as a helper during the updateWeights phase.
void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations, double sensor_range) 
{
    for (int i = 0; i < observations.size(); i++) {
	double lowest_dist      = sensor_range ;
	int closest_landmark_id = -1;
	double obs_x            = observations[i].x;
	double obs_y            = observations[i].y;

	for (int j = 0; j < predicted.size(); j++) {
	    double pred_x = predicted[j].x;
	    double pred_y = predicted[j].y;
	    int pred_id   = predicted[j].id;
	    double current_dist = dist(obs_x, obs_y, pred_x, pred_y);

	    if (current_dist < lowest_dist) {
	        lowest_dist         = current_dist;
	        closest_landmark_id = pred_id;
                //cout << "dataAssociation = " << pred_id << endl;
	    }
	}
	observations[i].id = closest_landmark_id;
    }

}
void ParticleFilter::resample() {
    // Resamples particles with replacement with probability proportional to their weight.
    // Vector for new particles
    vector<Particle> new_particles (num_particles);
  
    // Use discrete distribution to return particles by weight
    random_device rd;
    default_random_engine gen(rd());
    for (int i = 0; i < num_particles; ++i) {
        discrete_distribution<int> index(weights.begin(), weights.end());
        new_particles[i] = particles[index(gen)%num_particles];
    }
    particles = new_particles;

}

    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates
Particle ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations, 
                                     const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
    //Clear the previous associations
    particle.associations.clear();
    particle.sense_x.clear();
    particle.sense_y.clear();

    particle.associations= associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;
    return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
