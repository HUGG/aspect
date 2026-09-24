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


#include <aspect/boundary_traction/scaled_lithostatic_pressure.h>
#include <aspect/initial_temperature/interface.h>
#include <aspect/initial_composition/interface.h>
#include <aspect/geometry_model/initial_topography_model/zero_topography.h>
#include <aspect/geometry_model/initial_topography_model/interface.h>
#include <aspect/gravity_model/interface.h>
#include <aspect/global.h>
#include <aspect/utilities.h>

#include <aspect/geometry_model/box.h>
#include <aspect/geometry_model/two_merged_boxes.h>

namespace
{
  // Parse "boundary_name : v1, v2, ... [; boundary_name : v1, v2, ... ]" into a
  // map from boundary name to the list of values given for it.
  std::map<std::string, std::vector<double>>
  parse_named_double_lists (const std::string &input, const std::string &parameter_name)
  {
    std::map<std::string, std::vector<double>> result;
    for (const auto &entry : dealii::Utilities::split_string_list(input, ';'))
      {
        const std::size_t colon = entry.find(':');
        AssertThrow(colon != std::string::npos,
                    dealii::ExcMessage("Invalid format for '" + parameter_name + "': "
                                       "each entry must be 'boundary_name : value[, value...]'."));
        std::string bname = entry.substr(0, colon);
        bname.erase(0, bname.find_first_not_of(" \t"));
        bname.erase(bname.find_last_not_of(" \t") + 1);

        result[bname] = dealii::Utilities::string_to_double(dealii::Utilities::split_string_list(entry.substr(colon + 1)));
      }
    return result;
  }

  // Parse "boundary_name : value [; boundary_name : value ...]" into a map from
  // boundary name to the (trimmed, unparsed) value string given for it.
  std::map<std::string, std::string>
  parse_named_strings (const std::string &input, const std::string &parameter_name)
  {
    std::map<std::string, std::string> result;
    for (const auto &entry : dealii::Utilities::split_string_list(input, ';'))
      {
        const std::size_t colon = entry.find(':');
        AssertThrow(colon != std::string::npos,
                    dealii::ExcMessage("Invalid format for '" + parameter_name + "': "
                                       "each entry must be 'boundary_name : value'."));
        std::string bname = entry.substr(0, colon);
        bname.erase(0, bname.find_first_not_of(" \t"));
        bname.erase(bname.find_last_not_of(" \t") + 1);
        std::string value = entry.substr(colon + 1);
        value.erase(0, value.find_first_not_of(" \t"));
        value.erase(value.find_last_not_of(" \t") + 1);

        result[bname] = value;
      }
    return result;
  }
}

namespace aspect
{
  namespace BoundaryTraction
  {

    template <int dim>
    void
    ScaledLithostaticPressure<dim>::initialize()
    {
      // Ensure the scaled lithostatic pressure traction boundary conditions are used,
      // and register for which boundary indicators these conditions are set.
      std::set<types::boundary_id> traction_bi;
      unsigned int i=0;
      for (const auto &plugin : this->get_boundary_traction_manager().get_active_plugins())
        {
          if (plugin.get() == this)
            traction_bi.insert(this->get_boundary_traction_manager().get_active_plugin_boundary_indicators()[i]);

          ++i;
        }
      AssertThrow(traction_bi.empty() == false,
                  ExcMessage("Did not find any boundary indicators for the scaled lithostatic pressure plugin."));

      // Translate the symbolic boundary names of all keyed parameters to IDs.
      std::map<types::boundary_id, Point<dim>> point_per_boundary_id;
      for (const auto &entry : representative_point_per_boundary_name)
        point_per_boundary_id[this->get_geometry_model().translate_symbolic_boundary_name_to_id(entry.first)] = entry.second;

      std::map<types::boundary_id, double> percentage_per_boundary_id;
      for (const auto &entry : perturbation_percentage_per_boundary_name)
        percentage_per_boundary_id[this->get_geometry_model().translate_symbolic_boundary_name_to_id(entry.first)] = entry.second;

      for (const auto &name : perturbation_percentage_function_boundary_names)
        perturbation_percentage_function_boundary_ids.insert(this->get_geometry_model().translate_symbolic_boundary_name_to_id(name));

      for (const types::boundary_id bid : traction_bi)
        {
          AssertThrow(point_per_boundary_id.count(bid) != 0,
                      ExcMessage("The 'scaled lithostatic pressure' plugin requires a 'Representative point' "
                                 "entry for every boundary it is applied to."));
          const bool uses_function = perturbation_percentage_function_boundary_ids.count(bid) != 0;
          AssertThrow(uses_function || percentage_per_boundary_id.count(bid) != 0,
                      ExcMessage("The 'scaled lithostatic pressure' plugin requires a 'Pressure perturbation "
                                 "percentage' entry (a number or 'function') for every boundary it is applied to."));

          const std::pair<std::vector<double>, double> profile = compute_pressure_profile(point_per_boundary_id.at(bid));
          pressure_profiles[bid]    = profile.first;
          delta_z_per_boundary[bid] = profile.second;

          if (!uses_function)
            perturbation_factor_per_boundary[bid] = 1.0 + percentage_per_boundary_id.at(bid) / 100.0;
        }
    }


    template <int dim>
    std::pair<std::vector<double>, double>
    ScaledLithostaticPressure<dim>::compute_pressure_profile (Point<dim> representative_point) const
    {
      // Adapted from the "initial lithostatic pressure" boundary traction plugin,
      // restricted to Cartesian box-type geometries.

      AssertThrow(Plugins::plugin_type_matches<const GeometryModel::Box<dim>> (this->get_geometry_model()) ||
                  Plugins::plugin_type_matches<const GeometryModel::TwoMergedBoxes<dim>> (this->get_geometry_model()),
                  ExcMessage("The 'scaled lithostatic pressure' boundary traction plugin only supports "
                             "the 'box' and 'box with lithosphere boundary indicators' geometry models."));

      AssertThrow(this->get_geometry_model().point_is_in_domain(representative_point),
                  ExcMessage("The reference point does not lie within the domain."));

      const unsigned int n_compositional_fields = this->n_compositional_fields();

      std::vector<double> pressure(n_points, -1.0);
      pressure[0] = this->get_surface_pressure();

      // Get the origin and extent of the domain in the depth direction.
      double origin_z = 0.;
      double extent_z = 0.;
      if (Plugins::plugin_type_matches<const GeometryModel::Box<dim>> (this->get_geometry_model()))
        {
          const GeometryModel::Box<dim> &gm = Plugins::get_plugin_as_type<const GeometryModel::Box<dim>>(this->get_geometry_model());
          origin_z = gm.get_origin()[dim-1];
          extent_z = gm.get_extents()[dim-1];
        }
      else
        {
          const GeometryModel::TwoMergedBoxes<dim> &gm = Plugins::get_plugin_as_type<const GeometryModel::TwoMergedBoxes<dim>>(this->get_geometry_model());
          origin_z = gm.get_origin()[dim-1];
          extent_z = gm.get_extents()[dim-1];
        }

      // Place the representative point at the (undeformed) top surface.
      representative_point[dim-1] = origin_z + extent_z;
      const double depth_extent = extent_z;

      // If present, retrieve initial topography at the reference point.
      double topo = 0.;
      if (!Plugins::plugin_type_matches<const InitialTopographyModel::ZeroTopography<dim>>(this->get_initial_topography_model()))
        {
          Point<dim-1> surface_point;
          for (unsigned int d=0; d<dim-1; ++d)
            surface_point[d] = representative_point[d];
          topo = this->get_initial_topography_model().value(surface_point);
        }

      // The spacing of the depth profile at the location of the representative point.
      const double delta_z = (depth_extent + topo) / (n_points-1);

      // Set up the input for the density function of the material model.
      typename MaterialModel::Interface<dim>::MaterialModelInputs in(1, n_compositional_fields);
      typename MaterialModel::Interface<dim>::MaterialModelOutputs out(1, n_compositional_fields);
      in.requested_properties = MaterialModel::MaterialProperties::density;

      in.position[0] = representative_point;

      // We need the initial temperature at this point
      in.temperature[0] = this->get_initial_temperature_manager().initial_temperature(in.position[0]);

      // and the surface pressure.
      in.pressure[0] = pressure[0];

      // Then the compositions at this point.
      for (unsigned int c=0; c<n_compositional_fields; ++c)
        in.composition[0][c] = this->get_initial_composition_manager().initial_composition(in.position[0], c);

      // We set all entries of the velocity vector to zero since this is the lithostatic case.
      in.velocity[0] = Tensor<1,dim> ();

      // Evaluate the material model to get the density.
      this->get_material_model().evaluate(in, out);
      const double density0 = out.densities[0];

      // Get the magnitude of gravity. We assume that gravity always points
      // along the depth direction.
      const double gravity0 = this->get_gravity_model().gravity_vector(in.position[0]).norm();

      // Now integrate pressure downward using trapezoidal integration
      // p'(z) = rho(p,c,T) * |g| * delta_z
      double sum = delta_z * 0.5 * density0 * gravity0;

      for (unsigned int i=1; i<n_points; ++i)
        {
          // Decrease z coordinate with depth increment.
          representative_point[dim-1] -= delta_z;
          in.position[0] = representative_point;

          // Retrieve the initial temperature at this point.
          in.temperature[0] = this->get_initial_temperature_manager().initial_temperature(in.position[0]);
          // and use the previous pressure.
          in.pressure[0] = pressure[i-1];

          // Retrieve the compositions at this point.
          for (unsigned int c=0; c<n_compositional_fields; ++c)
            in.composition[0][c] = this->get_initial_composition_manager().initial_composition(in.position[0], c);

          in.velocity[0] = Tensor<1,dim> ();

          // Evaluate the material model to get the density at the current point.
          this->get_material_model().evaluate(in, out);
          const double density = out.densities[0];

          // Get the magnitude of gravity.
          const double gravity = this->get_gravity_model().gravity_vector(in.position[0]).norm();

          // Trapezoid integration.
          pressure[i] = sum + delta_z * 0.5 * density * gravity;
          sum += delta_z * density * gravity;
        }

      Assert (*std::min_element (pressure.begin(), pressure.end()) >=
              -std::numeric_limits<double>::epsilon() * pressure.size(),
              ExcInternalError());

      return std::make_pair(pressure, delta_z);
    }


    template <int dim>
    Tensor<1,dim>
    ScaledLithostaticPressure<dim>::
    boundary_traction (const types::boundary_id boundary_indicator,
                       const Point<dim> &position,
                       const Tensor<1,dim> &normal_vector) const
    {
      // Set the normal component to the (percentage-scaled) lithostatic
      // pressure, the rest of the traction components are left set to zero.
      const double scaling_factor = perturbation_percentage_function_boundary_ids.count(boundary_indicator) != 0
                                       ? 1.0 + perturbation_percentage_function.value(position) / 100.0
                                       : perturbation_factor_per_boundary.at(boundary_indicator);
      const double pressure = interpolate_pressure(position, boundary_indicator) * scaling_factor;

      return -pressure * normal_vector;
    }

    template <int dim>
    void
    ScaledLithostaticPressure<dim>::update()
    {
      Interface<dim>::update();

      if (this->convert_output_to_years())
        perturbation_percentage_function.set_time (this->get_time() / year_in_seconds);
      else
        perturbation_percentage_function.set_time (this->get_time());
    }

    template <int dim>
    double
    ScaledLithostaticPressure<dim>::
    interpolate_pressure (const Point<dim> &p, const types::boundary_id bid) const
    {
      const std::vector<double> &pressure = pressure_profiles.at(bid);
      const double delta_z = delta_z_per_boundary.at(bid);

      // The depth at which we need the pressure.
      const double z = this->get_geometry_model().depth(p);

      // Check that depth does not exceed the maximal depth.
      if (z >= this->get_geometry_model().maximal_depth())
        {
          Assert (z <= this->get_geometry_model().maximal_depth() + delta_z,
                  ExcInternalError());
          // Return deepest (last) pressure
          return pressure.back();
        }

      const unsigned int i = static_cast<unsigned int>(z/delta_z);
      // If mesh deformation is allowed, the depth can become
      // negative. However, the returned depth is capped at 0
      // by the geometry models and thus always positive.
      Assert ((z/delta_z) >= 0, ExcInternalError());
      Assert (i+1 < pressure.size(), ExcInternalError());

      // Now do the linear interpolation.
      const double d=1.0+i-z/delta_z;
      Assert ((d>=0) && (d<=1), ExcInternalError());

      return d*pressure[i]+(1-d)*pressure[i+1];
    }


    template <int dim>
    void
    ScaledLithostaticPressure<dim>::declare_parameters (ParameterHandler &prm)
    {
      prm.enter_subsection("Boundary traction model");
      {
        prm.enter_subsection("Scaled lithostatic pressure");
        {
          prm.declare_entry ("Representative point", "",
                             Patterns::Anything(),
                             "The point(s) where the pressure profile will be calculated, keyed by "
                             "symbolic boundary name. Every boundary this plugin is applied to must "
                             "have an entry. "
                             "Format: boundary\\_name : x, y [; boundary\\_name : x, y ...] (2d) or "
                             "boundary\\_name : x, y, z [; ...] (3d). Note that the coordinate related "
                             "to the depth ($y$ in 2d, $z$ in 3d) is not used. "
                             "Units: \\si{\\meter}.");
          prm.declare_entry("Number of integration points", "1000",
                            Patterns::Integer(0),
                            "The number of integration points over which we integrate the lithostatic pressure "
                            "downwards.");
          prm.declare_entry ("Pressure perturbation percentage", "",
                             Patterns::Anything(),
                             "The percentage(s) by which the lithostatic pressure normal traction is "
                             "increased (positive values) or decreased (negative values), keyed by "
                             "symbolic boundary name. Every boundary this plugin is applied to must "
                             "have an entry. Each value is either a constant number, or the keyword "
                             "'function', in which case the percentage for that boundary is evaluated "
                             "using the shared function declared in the 'Pressure perturbation "
                             "percentage function' subsection. "
                             "Format: boundary\\_name : percentage\\_or\\_function [; boundary\\_name : "
                             "percentage\\_or\\_function ...], e.g. 'left : 5; right : function'. "
                             "The prescribed traction is $-(1+p/100)\\ P_{lithostatic}$, where $p$ "
                             "is this percentage and $P_{lithostatic}$ is the lithostatic pressure. "
                             "Units: \\%.");

          prm.enter_subsection("Pressure perturbation percentage function");
          {
            Functions::ParsedFunction<dim>::declare_parameters (prm, 1);
          }
          prm.leave_subsection();
        }
        prm.leave_subsection();
      }
      prm.leave_subsection();
    }


    template <int dim>
    void
    ScaledLithostaticPressure<dim>::parse_parameters (ParameterHandler &prm)
    {
      unsigned int refinement;
      prm.enter_subsection("Mesh refinement");
      {
        refinement = prm.get_integer("Initial adaptive refinement") + prm.get_integer("Initial global refinement");
      }
      prm.leave_subsection();

      prm.enter_subsection("Boundary traction model");
      {
        prm.enter_subsection("Scaled lithostatic pressure");
        {
          n_points = prm.get_integer("Number of integration points");

          for (const auto &entry : parse_named_double_lists(prm.get("Representative point"), "Representative point"))
            {
              AssertThrow(entry.second.size() == dim,
                          ExcMessage("Representative point for boundary '" + entry.first +
                                     "' does not have the right dimensions."));
              Point<dim> pt;
              for (unsigned int d = 0; d < dim; ++d)
                pt[d] = entry.second[d];
              representative_point_per_boundary_name[entry.first] = pt;
            }

          for (const auto &entry : parse_named_strings(prm.get("Pressure perturbation percentage"), "Pressure perturbation percentage"))
            {
              if (entry.second == "function")
                perturbation_percentage_function_boundary_names.insert(entry.first);
              else
                perturbation_percentage_per_boundary_name[entry.first] = dealii::Utilities::string_to_double(entry.second);
            }

          prm.enter_subsection("Pressure perturbation percentage function");
          {
            try
              {
                perturbation_percentage_function.parse_parameters (prm);
              }
            catch (...)
              {
                std::cerr << "ERROR: FunctionParser failed to parse\n"
                          << "\t'Boundary traction model.Scaled lithostatic pressure.Pressure perturbation percentage function'\n"
                          << "with expression\n"
                          << "\t'" << prm.get("Function expression") << "'\n"
                          << "More information about the cause of the parse error \n"
                          << "is shown below.\n";
                throw;
              }
          }
          prm.leave_subsection();
        }
        prm.leave_subsection();
      }
      prm.leave_subsection();

      // Check that we have enough integration points for this mesh.
      AssertThrow(Utilities::pow(2u,refinement) <= n_points, ExcMessage("Not enough integration points for this resolution."));
    }

  }
}

// explicit instantiations
namespace aspect
{
  namespace BoundaryTraction
  {
    ASPECT_REGISTER_BOUNDARY_TRACTION_MODEL(ScaledLithostaticPressure,
                                            "scaled lithostatic pressure",
                                            "Implementation of a model in which the boundary "
                                            "traction is given in terms of a normal traction component "
                                            "set to the lithostatic pressure, scaled by a user-defined "
                                            "percentage: the prescribed traction is "
                                            "$-(1+p/100)\\ P_{lithostatic}$, where $p$ is the percentage "
                                            "set by 'Pressure perturbation percentage' for that boundary "
                                            "and $P_{lithostatic}$ is the lithostatic pressure. "
                                            "A positive percentage increases the compressive normal traction "
                                            "relative to the lithostatic reference state, a negative "
                                            "percentage decreases it (extension). "
                                            "\n\n"
                                            "The lithostatic pressure is calculated the same way as in the "
                                            "'initial lithostatic pressure' plugin, by integrating "
                                            "the pressure downward based on the initial composition "
                                            "and temperature along a user-specified representative depth "
                                            "profile (see 'Representative point'). Both 'Representative "
                                            "point' and 'Pressure perturbation percentage' are keyed by "
                                            "symbolic boundary name, and every boundary this plugin is "
                                            "applied to must have an entry in each. Instead of a constant "
                                            "number, the percentage for a boundary can also be given as "
                                            "the keyword 'function', in which case it is evaluated using "
                                            "the (space- and time-dependent) function shared by all "
                                            "boundaries set to 'function', declared in the 'Pressure "
                                            "perturbation percentage function' subsection. "
                                            "\n\n"
                                            "Unlike the 'initial lithostatic pressure' plugin, this plugin "
                                            "only supports the 'box' and 'box with lithosphere boundary "
                                            "indicators' geometry models, and does not support prescribing "
                                            "a constant pressure at the bottom boundary when initial "
                                            "topography is used. "
                                            "\n\n"
                                            "Gravity is expected to point along the depth direction.")
  }
}
