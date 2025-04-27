/**
 * @file MasterClass.h
 * @brief This file contains the definition of the MasterClass, which serves as the central
 *        class for managing and coordinating the functionality of all other components
 *        within the project.
 *
 * Responsibilities:
 * - Acts as the entry point for initializing and managing the project.
 * - Coordinates communication between different modules.
 * - Provides utility functions and shared resources for other components.
 * - Ensures modularity and scalability of the project architecture.
 *
 * This file is critical for maintaining the overall structure and functionality of the
 * project, making it the master class that ties all other files together.
 */

#ifndef TyNET_HPP
#define TyNET_HPP
#include <Eigen/Dense>
#include <iostream>


//create an instance of the TyNET class
class TyNET {
public:
    TyNET();
    ~TyNET();

private:

};



#endif