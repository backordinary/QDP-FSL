# https://github.com/fabianbauermarquart/quantum_sensor_optimization/blob/0c0c0abe3374f59af2f01eff02ee863bcd0b14c2/main.py
import csv
from typing import List

import matplotlib.pyplot as plt
import numpy as np
from docplex.mp.model import Model
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from qiskit import BasicAer
from qiskit.aqua import aqua_globals, QuantumInstance
from qiskit.aqua.algorithms import VQE
from qiskit.optimization import QuadraticProgram
from qiskit.optimization.algorithms import MinimumEigenOptimizer
from seaborn import diverging_palette
import multiprocessing as mp
from pathlib import Path

from utils.geometry import rotate_point, create_cubic_data, calculate_plane_normal, get_vector_angles, \
    sample_parallelogram_points, vector_intersects_plane, is_point_visible_by_sensor
from utils.models import Sensor, SensorType, Characteristic, FieldOfView
from utils.types import is_number, flatten_list, group_list
from tqdm import tqdm
import itertools

multiprocessing = False


def main():
    allowed_sensor_positions = {}
    occlusion_geometry = {}
    criticality_grid = []
    regions_of_interest_cubic = {}
    regions_of_interest_sector = {}

    # Read allowed sensor positions
    with open('Sensor_position/allowed_sensor_positions.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=';')

        row_count = 0
        for row in csv_reader:
            if row_count == 0:
                row_count += 1
            else:
                allowed_sensors = [SensorType[str] for str in row[13].lower().split(', ')]

                allowed_sensor_positions[row[0]] = {'coordinates': group_list(row[1:13], 3),
                                                    'plane_normal': calculate_plane_normal(group_list(row[1:13], 3)),
                                                    'allowed_sensors': allowed_sensors}
                row_count += 1

    # Read occlusion geometry
    with open('Sensor_position/occlusion_geometry.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=';')

        row_count = 0
        for row in csv_reader:
            if row_count == 0:
                row_count += 1
            else:
                row_entries = [v for v in row if is_number(v)]
                occlusion_geometry[row[0]] = {'coordinates': group_list(row_entries, 3)}
                row_count += 1

    # Read criticality grid
    scaling_factor = 1000

    with open('Sensor_position/criticallity_grid_0_5.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        row_count = 0

        for row in csv_reader:
            if row_count == 0:
                row_count += 1
            else:
                point = np.array(row, dtype=float)
                point[0:3] = point[0:3] * scaling_factor

                criticality_grid.append(point)
                row_count += 1

    criticality_grid = np.array(criticality_grid, dtype=float)
    criticality_grid_display = np.array(criticality_grid, dtype=float)
    criticality_grid_display = np.delete(criticality_grid_display,
                                         list(range(0, criticality_grid_display.shape[0], 2)), axis=0)
    criticality_grid_display = np.delete(criticality_grid_display,
                                         list(range(0, criticality_grid_display.shape[0], 2)), axis=0)
    criticality_grid_display = np.delete(criticality_grid_display,
                                         list(range(0, criticality_grid_display.shape[0], 2)), axis=0)
    criticality_grid_display = np.delete(criticality_grid_display,
                                         list(range(0, criticality_grid_display.shape[0], 2)), axis=0)
    criticality_grid_display = np.delete(criticality_grid_display,
                                         list(range(0, criticality_grid_display.shape[0], 2)), axis=0)

    # Read regions of interest (cubic)
    with open('Sensor_position/regions_of_interest_cubic.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=';')
        row_count = 0

        for row in csv_reader:
            if row_count == 0:
                row_count += 1
            else:
                start_point = np.array(row[1:4], dtype=float) * scaling_factor
                end_point = np.array(row[4:7], dtype=float) * scaling_factor

                regions_of_interest_cubic[row[0]] = {'start_point': start_point,
                                                     'end_point': end_point,
                                                     'rel_speed': float(row[7]),
                                                     'distance': float(row[8]),
                                                     'critical_index': float(row[9]),
                                                     'required_parameters': [v for v in row[10:13] if v != ''],
                                                     'speed': float(row[13]),
                                                     'env_types': [v for v in row[14:17] if v != '']}
                row_count += 1

    # Read regions of interest (spherical sector)
    with open('Sensor_position/regions_of_interest_sector.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=';')
        row_count = 0

        for row in csv_reader:
            if row_count == 0:
                row_count += 1
            else:
                center_point = np.array(row[1:4], dtype=float)
                radius_max = float(row[4])
                radius_min = float(row[5])
                yaw_angle = float(row[6])
                height = float(row[7])
                h_angle = float(row[8])

                regions_of_interest_sector[row[0]] = {'center_point': center_point,
                                                      'radius_max': radius_max,
                                                      'radius_min': radius_min,
                                                      'yaw_angle': yaw_angle,
                                                      'height': height,
                                                      'h_angle': h_angle,
                                                      'rel_speed': float(row[9]),
                                                      'distance': float(row[10]),
                                                      'critical_index': float(row[11]),
                                                      'required_parameters': [v for v in row[12:15] if v != ''],
                                                      'speed': float(row[15]),
                                                      'env_types': [v for v in row[16:19] if v != '']}

                regions_of_interest_sector[row[0]]['radius_max_point'] \
                    = (center_point + rotate_point(np.array([radius_max, 0, 0]), yaw_angle, 0,
                                                   h_angle)) * scaling_factor

                regions_of_interest_sector[row[0]]['radius_min_point'] \
                    = (center_point + rotate_point(np.array([radius_min, 0, 0]), yaw_angle, 0,
                                                   h_angle)) * scaling_factor

                row_count += 1

    # Print outs for plausability check
    # print(allowed_sensor_positions)
    # print(occlusion_geometry)
    # print(criticality_grid[0])
    # print(regions_of_interest_cubic)
    # print(regions_of_interest_sector)

    # Set up plot
    fig = plt.figure()
    ax = Axes3D(fig)

    # Car only
    ax.set_xlim3d(-10000, 5000)
    ax.set_ylim3d(-4000, 4000)
    ax.set_zlim3d(0, 4000)

    # Regions of interest
    ax.set_xlim3d(-75000, 125000)
    ax.set_ylim3d(-100000, 100000)
    ax.set_zlim3d(0, 100000)

    sensor_pos_collection = Poly3DCollection(
        [v['coordinates'] for v in allowed_sensor_positions.values()],
        linewidths=0.5, alpha=0.8, edgecolors='black', facecolors='blue')

    occlusion_geometry_collection = Poly3DCollection(
        [v['coordinates'] for v in occlusion_geometry.values()],
        linewidths=0.5, alpha=0.5, edgecolors='black', facecolors='green')

    regions_of_interest_cubic_collection = Poly3DCollection(
        flatten_list([create_cubic_data(v['start_point'], v['end_point']) for v in regions_of_interest_cubic.values()]),
        linewidths=0.5, edgecolors='black', alpha=0.3, facecolors='red')

    # ax.add_collection3d(occlusion_geometry_collection)
    ax.add_collection3d(sensor_pos_collection)
    ax.add_collection3d(regions_of_interest_cubic_collection)

    for value in regions_of_interest_sector.values():
        xs = [value['radius_min_point'][0], value['radius_max_point'][0]]
        ys = [value['radius_min_point'][1], value['radius_max_point'][1]]
        zs = [value['radius_min_point'][2], value['radius_max_point'][2]]

        ax.plot(xs, ys, zs)

    palette = diverging_palette(0, 255, sep=8, n=256)
    criticality_palette = [palette[int(v * 255)] for v in criticality_grid_display[:, 3]]

    ax.scatter(criticality_grid_display[:, 0], criticality_grid_display[:, 1], criticality_grid_display[:, 2],
               s=1,
               c=criticality_palette, alpha=0.5)

    # plt.show()

    # Basic sensors

    # Livox Mid-100
    lidar_characteristic = Characteristic(SensorType.lidar,
                                          FieldOfView(vertical_angle=38.4 * np.pi / 180,
                                                      horizontal_angle=98.4 * np.pi / 180,
                                                      fov_range=90 * 1000),
                                          price=1500.00)

    # Unspecified short range radar
    short_range_radar_characteristic = Characteristic(SensorType.radar,
                                                      FieldOfView(vertical_angle=80 * np.pi / 180,
                                                                  horizontal_angle=80 * np.pi / 180,
                                                                  fov_range=30 * 1000),
                                                      price=200.00)

    # Bosch front radar
    long_range_radar_characteristic = Characteristic(SensorType.radar,
                                                     FieldOfView(vertical_angle=6 * np.pi / 180,
                                                                 horizontal_angle=3 * np.pi / 180,
                                                                 fov_range=210 * 1000),
                                                     price=200.00)

    # Bosch multi purpose camera
    camera_characteristic = Characteristic(SensorType.camera,
                                           FieldOfView(vertical_angle=58 * np.pi / 180,
                                                       horizontal_angle=50 * np.pi / 180,
                                                       fov_range=500 * 1000),
                                           price=3000.00)

    # Bosch ultrasonic sensor
    ultrasonic_characteristic = Characteristic(SensorType.ultrasound,
                                               FieldOfView(vertical_angle=35 * np.pi / 180,
                                                           horizontal_angle=70 * np.pi / 180,
                                                           fov_range=5.5 * 1000),
                                               price=60.0)

    sensor_candidates = []

    # Calculate sensor candidates, including position and orientation
    if Path('sensor_candidates.csv').is_file():
        with open('sensor_candidates.csv') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')

            row_count = 0
            for row in csv_reader:
                if row_count == 0:
                    row_count += 1
                else:
                    sensor = Sensor(Characteristic(SensorType[row[1]],
                                                   FieldOfView(vertical_angle=row[2],
                                                               horizontal_angle=row[3],
                                                               fov_range=row[4]),
                                                   price=row[5]),
                                    position=np.array([row[6:9]]),
                                    orientation=(row[9], row[10]),
                                    dict={'orientation_vector': np.array(row[11:14]),
                                          'relative_orientation': (row[15], row[14]),
                                          'plane_normal': np.array(row[16:19])})
                    sensor_candidates.append(sensor)
                    row_count += 1
    else:
        # Place sensors on an even grid on all surfaces, considering all relevant orientations
        sensor_position_points = []

        for plane in allowed_sensor_positions.values():
            # Generate points on surface
            corners = plane['coordinates']
            sampled_positions = sample_parallelogram_points(corners, distance=150)
            sensor_position_points.extend(sampled_positions)

            # Generate sensor candidates
            allowed_sensor_types = plane['allowed_sensors']

            # Sensor angles
            vertical_angles = np.linspace(0, np.pi / 2, num=2, endpoint=True)  # 4
            horizontal_angles = np.linspace(0, 2 * np.pi, num=2, endpoint=False)  # 8

            angle_pairs = [(vertical_angles[0], horizontal_angles[0])]
            angle_pairs.extend(itertools.product(vertical_angles[1:], horizontal_angles))

            plane_normal = plane['plane_normal']  # rotate according to plane normal
            orientation_vectors = [rotate_point(plane_normal, z_angle=horizontal_angle, x_angle=vertical_angle)
                                   for horizontal_angle, vertical_angle in angle_pairs]

            # TODO: check resulting vectors
            orientation_angles = [get_vector_angles(normal) for normal in orientation_vectors]

            # Create sensor for every position and angle
            for position in sampled_positions:
                current_candidates = []

                for type in allowed_sensor_types:
                    applicable_sensors = []

                    for i, orientation in enumerate(orientation_angles):
                        if type == SensorType.lidar:
                            applicable_sensors.append(Sensor(lidar_characteristic, position, orientation))
                        elif type == SensorType.radar:
                            applicable_sensors.append(Sensor(short_range_radar_characteristic, position, orientation))
                            applicable_sensors.append(Sensor(long_range_radar_characteristic, position, orientation))
                        elif type == SensorType.camera:
                            applicable_sensors.append(Sensor(camera_characteristic, position, orientation))
                        elif type == SensorType.ultrasound:
                            applicable_sensors.append(Sensor(ultrasonic_characteristic, position, orientation))

                    for sensor in applicable_sensors:
                        sensor.dict = {'orientation_vector': orientation_vectors[i],
                                       'relative_orientation': angle_pairs[i],
                                       'plane_normal': plane_normal}

                    current_candidates.extend(applicable_sensors)

                # print('current_candidates (before)', len(current_candidates))

                # filter out according to occlusion geometry
                for i, candidate in enumerate(current_candidates):
                    for occlusion_plane in occlusion_geometry.values():
                        plane_coordinates = occlusion_plane['coordinates']

                        start_point = candidate.position
                        end_point = candidate.position \
                                    + candidate.characteristic.field_of_view.fov_range \
                                    * candidate.dict['orientation_vector']

                        if vector_intersects_plane(start_point, end_point,
                                                   plane_coordinates[0], calculate_plane_normal(plane_coordinates)):
                            current_candidates[i] = None
                            continue

                current_candidates = [candidate for candidate in current_candidates if candidate is not None]

                # print('current_candidates (filtered)', len(current_candidates))

                sensor_candidates.extend(current_candidates)

        # Save sensor candidates to file
        with open('sensor_candidates.csv', 'w') as f:
            writer = csv.writer(f)

            # write the header
            writer.writerow(['key',
                             'type',
                             'fov_vertical_angle', 'fov_horizontal_angle', 'fov_range',
                             'price',
                             'position_x', 'position_y', 'position_z',
                             'orientation_phi', 'orientation_theta',
                             'orientation_vector_x', 'orientation_vector_y', 'orientation_vector_z',
                             'relative_orientation_phi', 'relative_orientation_theta',
                             'plane_normal_x', 'plane_normal_y', 'plane_normal_z'])

            # write rows
            for i, candidate in enumerate(sensor_candidates):
                writer.writerow([i,
                                 candidate.characteristic.type.name,
                                 candidate.characteristic.field_of_view.vertical_angle,
                                 candidate.characteristic.field_of_view.horizontal_angle,
                                 candidate.characteristic.field_of_view.fov_range,
                                 candidate.characteristic.price,
                                 candidate.position[0], candidate.position[1], candidate.position[2],
                                 candidate.orientation[0], candidate.orientation[1],
                                 candidate.dict['orientation_vector'][0],
                                 candidate.dict['orientation_vector'][1],
                                 candidate.dict['orientation_vector'][2],
                                 candidate.dict['relative_orientation'][1], candidate.dict['relative_orientation'][0],
                                 candidate.dict['plane_normal'][0],
                                 candidate.dict['plane_normal'][1],
                                 candidate.dict['plane_normal'][2]])

        # Visualize all possible sensor positions
        sensor_position_points = np.array(sensor_position_points)

        ax.scatter(sensor_position_points[:, 0], sensor_position_points[:, 1], sensor_position_points[:, 2],
                   s=2.5, linewidths=0.5, edgecolors='black', c='white')

        # plt.show()

    print('Number of sensor candidates:', len(sensor_candidates))

    # Calculate coverage booleans for sensor candidates
    conjunction_coverage_grid = [False] * len(criticality_grid)
    coverage_grids = [[False] * len(criticality_grid)] * len(sensor_candidates)

    if Path('coverage.csv').exists():
        # Read coverage file if it already exists
        with open('coverage.csv') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')

            row_count = 0
            for row in csv_reader:
                if row_count == 0:
                    row_count += 1
                else:
                    criticality_point_idx = int(row[0])
                    sensor_idx = int(row[1])
                    coverage = (row[2] == 'True')

                    if coverage:
                        coverage_grids[sensor_idx][criticality_point_idx] = True
                        conjunction_coverage_grid[criticality_point_idx] = True

                    row_count += 1
    else:
        # Else, evaluate coverage for all critical points and sensor candidates
        if not multiprocessing:
            # Sequential processing
            with open('coverage.csv', 'w') as f:
                writer = csv.writer(f)

                # header
                writer.writerow(['criticality_point', 'sensor', 'coverage'])

                for j, point in tqdm(enumerate(criticality_grid), total=len(criticality_grid)):
                    for i, sensor in enumerate(sensor_candidates):
                        result = is_point_visible_by_sensor((sensor, point[0:3], i, j))
                        coverage = result[0]

                        if coverage:
                            coverage_grids[i][j] = True
                            conjunction_coverage_grid[j] = True

                        writer.writerow([j, i, coverage])
        else:
            # Parallel processing
            with mp.Pool(mp.cpu_count()) as pool:
                results = tqdm(pool.imap(is_point_visible_by_sensor, [(sensor, point[0:3], i, j)
                                                                      for i, sensor in enumerate(sensor_candidates)
                                                                      for j, point in enumerate(criticality_grid)],
                                         chunksize=len(sensor_candidates)),
                               total=len(sensor_candidates) * len(criticality_grid))

            for result in results:
                (coverage, i, j) = result
                if coverage:
                    coverage_grids[i][j] = True
                conjunction_coverage_grid[j] = True

    def evaluate_coverage(region_of_interest_grid: np.ndarray, coverage_booleans: List[bool]) -> float:
        """
        Complete evaluation of the covered regions of interest: Ratio of the RoI grid covered by the sensor setup.

        Additionally, all regions of interest with a critical index above c_i ≥ 0.7 need to be covered by
        at least two sensors of a different type.

        :param region_of_interest_grid: region of interest grid point weights.
        :param coverage_booleans: 1 - grid point is covered by sensor setup, 0 - otherwise.
        :return: Coverage volume.
        """
        return np.sum(region_of_interest_grid[:, 3] * np.array(coverage_booleans, dtype=int)) \
               / np.sum(region_of_interest_grid[:, 3])

    print('Coverage for all sensor candidates combined:',
          evaluate_coverage(criticality_grid, conjunction_coverage_grid))

    # Quantum combinatorial optimization using a quadratic program.
    sample = 4

    sensor_candidates = sensor_candidates[:sample]
    coverage_grids = coverage_grids[:sample]

    print('Downsampled number of sensor candidates:', len(sensor_candidates))

    list_of_subsets = []

    for i, candidate in enumerate(sensor_candidates):
        subset = []

        # get index of every point seen by this sensor candidate
        for j in range(len(coverage_grids[i])):
            if coverage_grids[i][j]:
                subset.append(j)

        list_of_subsets.append(subset)

    n = len(criticality_grid)
    N = len(list_of_subsets)

    p = [float(sensor_candidates[i].characteristic.price) for i in range(N)]

    A = 1.0
    B = 0.0001
    criticality_threshold = 0.7

    # build model with docplex
    print('Building model with docplex...')

    mdl = Model()
    x = [mdl.binary_var() for _ in range(N)]

    objective = A * mdl.sum((1 - mdl.sum(x[i] for i in range(N)
                                         if alpha in list_of_subsets[i])) ** 2
                            + (1 - mdl.sum(x[i] for i in range(N)
                                           if alpha in list_of_subsets[i]
                                           and criticality_grid[alpha, 3] > criticality_threshold))
                            for alpha in range(n)) \
                + B * mdl.sum(x[i] * p[i] for i in range(N))

    mdl.minimize(objective)

    # convert to Qiskit's quadratic program
    qp = QuadraticProgram()
    qp.from_docplex(mdl)

    print(qp)

    print(qp.to_ising())

    aqua_globals.random_seed = 10598
    quantum_instance = QuantumInstance(BasicAer.get_backend('qasm_simulator'),
                                       seed_simulator=aqua_globals.random_seed,
                                       seed_transpiler=aqua_globals.random_seed)

    vqe = VQE(quantum_instance=quantum_instance)

    optimizer = MinimumEigenOptimizer(min_eigen_solver=vqe)

    result = optimizer.solve(qp)

    print(result)

    print(result.variables_dict)

    optimal_configuration_coverage_grid = [False] * len(criticality_grid)

    for sensor_idx in range(len(sensor_candidates)):
        if result.variables_dict[f'x{sensor_idx}'] != 1.0:
            continue

        for criticality_point_idx in range(len(coverage_grids[0])):
            if coverage_grids[sensor_idx][criticality_point_idx]:
                optimal_configuration_coverage_grid[criticality_point_idx] = True

    optimal_coverage = evaluate_coverage(criticality_grid, optimal_configuration_coverage_grid)

    print('Coverage for optimal configuration:', optimal_coverage)


if __name__ == "__main__":
    main()
