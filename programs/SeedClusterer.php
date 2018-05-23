<?php

include dirname(__DIR__) . '/vendor/autoload.php';

use Rubix\Engine\Pipeline;
use Rubix\Engine\GridSearch;
use Rubix\Engine\PersistentModel;
use Rubix\Engine\Datasets\Labeled;
use Rubix\Engine\Clusterers\KMeans;
use Rubix\Engine\Datasets\Unlabeled;
use Rubix\Engine\CrossValidation\KFold;
use Rubix\Engine\Metrics\Distance\Euclidean;
use Rubix\Engine\Metrics\Validation\Homogeneity;
use Rubix\Engine\CrossValidation\ReportGenerator;
use Rubix\Engine\Transformers\NumericStringConverter;
use Rubix\Engine\CrossValidation\Reports\ContingencyTable;
use League\Csv\Reader;

echo '╔═════════════════════════════════════════════════════╗' . "\n";
echo '║                                                     ║' . "\n";
echo '║ Seed Clusterer using K Means                        ║' . "\n";
echo '║                                                     ║' . "\n";
echo '╚═════════════════════════════════════════════════════╝' . "\n";

$reader = Reader::createFromPath(dirname(__DIR__) . '/datasets/seeds.csv')
    ->setDelimiter(',')->setEnclosure('"')->setHeaderOffset(0);

$samples = iterator_to_array($reader->getRecords([
    'area', 'perimeter', 'compactness', 'length_of_kernel', 'width_of_kernel',
    'asymmetry_coefficient', 'length_of_kernel_grove',
]));

$labels = iterator_to_array($reader->fetchColumn('class'));

$dataset = new Labeled($samples, $labels);

$estimator = new Pipeline(new KMeans(3), [
    new NumericStringConverter(),
]);

$validator = new KFold(new Homogeneity(), 10);

$report = new ReportGenerator(new ContingencyTable(), 0.2);

var_dump($validator->score($estimator, $dataset));

var_dump($report->generate($estimator, $dataset));

$model = new PersistentModel($estimator);

$model->save(dirname(__DIR__) . '/models/seed_detector.model');
