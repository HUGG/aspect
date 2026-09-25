/*
  Copyright (C) 2011 - 2024 by the authors of the ASPECT code.

  This file is part of ASPECT.

  ASPECT is free software; you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation; either version 2, or (at your option)
  any later version.

  ASPECT is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with ASPECT; see the file LICENSE.  If not see
  <http://www.gnu.org/licenses/>.
*/


#ifndef _aspect_boundary_traction_scaled_lithostatic_pressure_h
#define _aspect_boundary_traction_scaled_lithostatic_pressure_h

#include <aspect/boundary_traction/interface.h>
#include <aspect/simulator_access.h>
#include <deal.II/base/parsed_function.h>
#include <map>
#include <set>


namespace aspect
{
  namespace BoundaryTraction
  {
    /**
     * A class that implements traction boundary conditions by prescribing
     * the initial lithostatic pressure, scaled by a user-defined percentage,
     * as the normal traction component. This can be used to drive
     * compression (positive percentage) or extension (negative percentage)
     * relative to the lithostatic reference state, e.g. on opposite sides of
     * a box model.
     *
     * The lithostatic pressure profile is computed the same way as the
     * "initial lithostatic pressure" plugin (vertical trapezoidal
     * integration of the initial density along a representative column),
     * but this implementation only supports Cartesian box-type geometries
     * (the "box" and "box with lithosphere boundary indicators" geometry
     * models).
     *
     * @ingroup BoundaryTractions
     */
    template <int dim>
    class ScaledLithostaticPressure : public Interface<dim>, public SimulatorAccess<dim>
    {
      public:

        /**
         * Initialization function. Because this function is called after
         * initializing the SimulatorAccess, all of the necessary information
         * is available to calculate the pressure profile based on the initial
         * temperature and pressure conditions.
         */
        void initialize () override;


        /**
         * Return the boundary traction as a function of position. The
         * (outward) normal vector to the domain is also provided as
         * a second argument.
         */
        Tensor<1,dim>
        boundary_traction (const types::boundary_id boundary_indicator,
                           const Point<dim> &position,
                           const Tensor<1,dim> &normal_vector) const override;


        /**
         * Updates the function time of the perturbation percentage function,
         * if used, to the current simulation time.
         */
        void
        update () override;


        /**
         * Declare the parameters this class takes through input files.
         */
        static
        void
        declare_parameters (ParameterHandler &prm);


        /**
         * Read the parameters this class declares from the parameter file.
         */
        void
        parse_parameters (ParameterHandler &prm) override;

      private:

        /**
         * The number of integration points per profile.
         */
        unsigned int n_points;

        /**
         * Representative points keyed by symbolic boundary name. Every boundary
         * this plugin is applied to must have an entry. The depth coordinate is
         * ignored.
         */
        std::map<std::string, Point<dim>> representative_point_per_boundary_name;

        /**
         * Perturbation percentages keyed by symbolic boundary name, for boundaries
         * using a constant value. Every boundary this plugin is applied to must
         * have an entry either here or in
         * perturbation_percentage_function_boundary_names.
         */
        std::map<std::string, double> perturbation_percentage_per_boundary_name;

        /**
         * Symbolic names of boundaries for which the perturbation percentage is
         * given by perturbation_percentage_function instead of a constant value.
         */
        std::set<std::string> perturbation_percentage_function_boundary_names;

        /**
         * The (shared) function used to evaluate the perturbation percentage for
         * all boundaries listed in perturbation_percentage_function_boundary_names.
         * Declared in the "Pressure perturbation percentage function" subsection.
         */
        Functions::ParsedFunction<dim> perturbation_percentage_function;

        /**
         * Computed lithostatic pressure profiles, one entry per boundary indicator.
         */
        std::map<types::boundary_id, std::vector<double>> pressure_profiles;

        /**
         * Depth spacing between integration points, one entry per boundary indicator.
         */
        std::map<types::boundary_id, double> delta_z_per_boundary;

        /**
         * The factor (1 + percentage/100) the interpolated lithostatic pressure
         * is multiplied with, one entry per boundary indicator using a constant
         * perturbation percentage.
         */
        std::map<types::boundary_id, double> perturbation_factor_per_boundary;

        /**
         * Boundary indicators for which the perturbation percentage is evaluated
         * through perturbation_percentage_function instead of
         * perturbation_factor_per_boundary.
         */
        std::set<types::boundary_id> perturbation_percentage_function_boundary_ids;

        /**
         * Compute the lithostatic pressure profile along the column defined by
         * the lateral coordinates of representative_point (depth coordinate ignored).
         * Returns {pressure_vector, delta_z}. Only Cartesian box-type geometries
         * ("box" and "box with lithosphere boundary indicators") are supported.
         */
        std::pair<std::vector<double>, double>
        compute_pressure_profile (Point<dim> representative_point) const;

        /**
         * Interpolate the stored pressure profile for boundary bid at position p.
         */
        double interpolate_pressure (const Point<dim> &p,
                                     types::boundary_id bid) const;
    };
  }
}


#endif
