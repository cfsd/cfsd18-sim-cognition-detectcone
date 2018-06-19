/**
 * Copyright (C) 2017 Chalmers Revere
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License
 * as published by the Free Software Foundation; either version 2
 * of the License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301,
 * USA.
 */

#ifndef OPENDLV_SIM_CFSD18_PERCEPTION_DETECTCONE_HPP
#define OPENDLV_SIM_CFSD18_PERCEPTION_DETECTCONE_HPP

#include "cluon-complete.hpp"
#include <opendlv-standard-message-set.hpp>

#include <cmath>
#include <Eigen/Dense>

class DetectCone {
 public:
  DetectCone(std::map<std::string, std::string> commandlineArguments, cluon::OD4Session &od4);
  DetectCone(DetectCone const &) = delete;
  DetectCone &operator=(DetectCone const &) = delete;
  ~DetectCone();
  void nextContainer(cluon::data::Envelope &);
  void body();

 private:
  cluon::OD4Session &m_od4;
  int m_senderStamp;
  float m_detectRange;
  float m_detectWidth;
  bool m_fakeSlamActivated;
  int m_nConesFakeSlam;
  float m_startX;
  float m_startY;
  float m_startHeading;
  std::string m_filename;
  float m_heading;
  Eigen::ArrayXXf m_location;
  Eigen::ArrayXXf m_leftCones;
  Eigen::ArrayXXf m_rightCones;
  Eigen::ArrayXXf m_smallCones;
  Eigen::ArrayXXf m_bigCones;
  bool m_orangeVisibleInSlam;
  std::mutex m_locationMutex;
  const double RAD2DEG = 57.295779513082325; // 1.0 / DEG2RAD

  void setUp();
  void tearDown();
  void readMap(std::string);
  Eigen::ArrayXXf simConeDetectorBox(Eigen::ArrayXXf, Eigen::ArrayXXf, float, float, float);
  Eigen::ArrayXXf simConeDetectorSlam(Eigen::ArrayXXf, Eigen::ArrayXXf, float, int);
  void sendMatchedContainer(Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd);
  void sendCone(opendlv::logic::sensation::Point, cluon::data::TimeStamp, int, int);
  void Cartesian2Spherical(double, double, double, opendlv::logic::sensation::Point &);
};

#endif
