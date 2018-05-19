<?php

include dirname(__DIR__) . '/vendor/autoload.php';

use Rubix\Engine\Estimators\Ridge;
use Rubix\Engine\Datasets\Supervised;
use Rubix\Engine\Metrics\Validation\RSquared;
use Rubix\Engine\Estimators\Wrappers\Pipeline;
use Rubix\Engine\ModelSelection\CrossValidator;
use Rubix\Engine\Transformers\MinMaxNormalizer;
use Rubix\Engine\Transformers\NumericStringConverter;
use Rubix\Engine\Transformers\DensePolynomialExpander;
use Rubix\Engine\ModelSelection\Reports\RegressionAnalysis;
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

$dataset = new Supervised($samples, $labels);

$estimator = new Pipeline(new Ridge(0.05), [
    new NumericStringConverter(),
    new MinMaxNormalizer(),
    new DensePolynomialExpander(3),
]);

$validator = new CrossValidator(new RSquared());

var_dump($validator->validate($estimator, $dataset->fold(10)));
