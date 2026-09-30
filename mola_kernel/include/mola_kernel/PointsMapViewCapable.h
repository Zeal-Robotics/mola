/*               _
 _ __ ___   ___ | | __ _
| '_ ` _ \ / _ \| |/ _` | Modular Optimization framework for
| | | | | | (_) | | (_| | Localization and mApping (MOLA)
|_| |_| |_|\___/|_|\__,_| https://github.com/MOLAorg/mola

 Copyright (C) 2018-2026 Jose Luis Blanco, University of Almeria,
                         and individual contributors.
 SPDX-License-Identifier: GPL-3.0
 See LICENSE for full license information.
*/

/**
 * @file   PointsMapViewCapable.h
 * @brief  Virtual interface for metric maps that can render themselves as a points map.
 */
#pragma once

#include <mrpt/maps/CMetricMap.h>
#include <mrpt/maps/CPointsMap.h>
#include <mrpt/maps/CSimplePointsMap.h>

namespace mola
{
/** Mixin interface for metric map classes that can render their contents as a points map,
 *  for publishing or visualization by callers that do not know the concrete map class.
 *
 *  MRPT 3 retired the `mrpt::maps::CMetricMap::getAsSimplePointsMap()` virtual in favour of the
 *  free function mrpt::maps::asPointsMap(), which only answers for maps that already are points
 *  maps. This interface restores the extension point for every other map class.
 */
class PointsMapViewCapable
{
 public:
  PointsMapViewCapable()                                       = default;
  PointsMapViewCapable(const PointsMapViewCapable&)            = default;
  PointsMapViewCapable& operator=(const PointsMapViewCapable&) = default;
  PointsMapViewCapable(PointsMapViewCapable&&)                 = default;
  PointsMapViewCapable& operator=(PointsMapViewCapable&&)      = default;
  virtual ~PointsMapViewCapable()                              = default;

  /** A points map holding the map's current contents, or nullptr if it has none to offer.
   *  The pointer is a non-owning observer, valid until the map is modified or destroyed, or
   *  until this method is called again.
   */
  [[nodiscard]] virtual const mrpt::maps::CSimplePointsMap* getAsSimplePointsMap() const = 0;
};

/** A points map view of `map`: its own getAsSimplePointsMap() if it implements
 *  PointsMapViewCapable, otherwise whatever mrpt::maps::asPointsMap() returns. nullptr if
 *  neither has one. Same lifetime as PointsMapViewCapable::getAsSimplePointsMap().
 */
[[nodiscard]] inline const mrpt::maps::CPointsMap* asPointsMap(const mrpt::maps::CMetricMap& map)
{
  if (const auto* viewable = dynamic_cast<const PointsMapViewCapable*>(&map); viewable)
  {
    return viewable->getAsSimplePointsMap();
  }
  return mrpt::maps::asPointsMap(map);
}

}  // namespace mola
