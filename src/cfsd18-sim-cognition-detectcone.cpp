/*
 * Copyright (C) 2018  Christian Berger
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

#include "cluon-complete.hpp"
#include "opendlv-standard-message-set.hpp"
#include "detectcone.hpp"
#include <Eigen/Dense>
#include <cstdint>
#include <tuple>
#include <utility>
#include <iostream>
#include <string>
#include <thread>

int32_t main(int32_t argc, char **argv) {
  int32_t retCode{0};
  std::map<std::string, std::string> commandlineArguments = cluon::getCommandlineArguments(argc, argv);
  if (commandlineArguments.size()<=0) {
    std::cerr << argv[0] << " is a module simulating a perception system in the CFSD18 project." << std::endl;
    std::cerr << "Usage:   " << argv[0] << " --cid=<OpenDaVINCI session> [--id=<Identifier in case of simulated units>] [--verbose] [--freq] [Module specific parameters....]" << std::endl;
    std::cerr << "Example: " << argv[0] << "--cid=111 --id=231 --freq=20 --startX=0 --startY=0 --startHeading=3.14 [more...]" <<  std::endl;
    retCode = 1;
  } else {

    const float freq{(commandlineArguments["freq"].size() != 0) ? static_cast<float>(std::stof(commandlineArguments["freq"])) : (50.0f)};

    // Interface to a running OpenDaVINCI session.
    cluon::OD4Session od4{static_cast<uint16_t>(std::stoi(commandlineArguments["cid"]))};
    DetectCone detectcone(commandlineArguments, od4);

    auto bodyEnvelope{[&detecter = detectcone]() -> bool
    {
      detecter.body();
      return true;
    }
    };
    auto dataEnvelope{[&detecter = detectcone](cluon::data::Envelope &&envelope)
      {
          detecter.nextContainer(envelope);
      }
    };

    od4.dataTrigger(opendlv::sim::Frame::ID(),dataEnvelope);
    od4.timeTrigger(freq, bodyEnvelope);

  }
  return retCode;
}
