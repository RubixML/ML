<?php

include dirname(__DIR__) . '/vendor/autoload.php';

use Rubix\Engine\Pipeline;
use Rubix\Engine\Regressors\Ridge;
use Rubix\Engine\Datasets\Labeled;
use Rubix\Engine\CrossValidation\KFold;
use Rubix\Engine\Metrics\Validation\RSquared;
use Rubix\Engine\Transformers\MinMaxNormalizer;
use Rubix\Engine\CrossValidation\ReportGenerator;
use Rubix\Engine\Transformers\NumericStringConverter;
use Rubix\Engine\Transformers\DensePolynomialExpander;
use Rubix\Engine\CrossValidation\Reports\RegressionAnalysis;
use League\Csv\Reader;

echo '╔═════════════════════════════════════════════════════╗' . "\n";
echo '║                                                     ║' . "\n";
echo '║ Wine Tester using a Ridge Regressor                 ║' . "\n";
echo '║                                                     ║' . "\n";
echo '╚═════════════════════════════════════════════════════╝' . "\n";

echo  "\n";

$reader = Reader::createFromPath(dirname(__DIR__) . '/datasets/winequality-red.csv')
    ->setDelimiter(';')->setEnclosure('"')->setHeaderOffset(0);

$samples = iterator_to_array($reader->getRecords([
    'fixed_acidity', 'volatile_acidity', 'citric_acid', 'residual_sugar',
    'chlorides', 'free_sulfur_dioxide', 'total_sulfur_dioxide', 'density', 'pH',
    'sulphates', 'alcohol',
]));

$labels = iterator_to_array($reader->fetchColumn('quality'));

$dataset = new Labeled($samples, $labels);

$estimator = new Pipeline(new Ridge(0.5), [
    new NumericStringConverter(),
    new MinMaxNormalizer(),
    new DensePolynomialExpander(3),
]);

$validator = new KFold(new RSquared(), 10);

$report = new ReportGenerator(new RegressionAnalysis(), 0.2);

var_dump($validator->score($estimator, $dataset));

var_dump($report->generate($estimator, $dataset));
