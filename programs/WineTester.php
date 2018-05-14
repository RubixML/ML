<?php

include dirname(__DIR__) . '/vendor/autoload.php';

use Rubix\Engine\Ridge;
use Rubix\Engine\Pipeline;
use Rubix\Engine\Prototype;
use Rubix\Engine\GridSearch;
use Rubix\Engine\Metrics\RSquared;
use Rubix\Engine\Datasets\Supervised;
use Rubix\Engine\Reports\RegressionAnalysis;
use Rubix\Engine\Transformers\MinMaxNormalizer;
use Rubix\Engine\Transformers\DensePolynomialExpander;
use League\Csv\Reader;

echo '╔═════════════════════════════════════════════════════╗' . "\n";
echo '║                                                     ║' . "\n";
echo '║ Wine Tester using a Ridge Regressor                 ║' . "\n";
echo '║                                                     ║' . "\n";
echo '╚═════════════════════════════════════════════════════╝' . "\n";

echo  "\n";

$dataset = Reader::createFromPath(dirname(__DIR__) . '/datasets/winequality-red.csv')->setDelimiter(';')->getRecords();

$dataset = Supervised::fromIterator($dataset);

list($training, $testing) = $dataset->randomize()->split(0.8);

$estimator = new Prototype(new Pipeline(new GridSearch(Ridge::class, [[0, 0.5, 1, 2, 3, 5]], new RSquared()), [
    new MinMaxNormalizer(),
    new DensePolynomialExpander(3),
]), [
    new RegressionAnalysis(),
]);

$estimator->train($training);

var_dump($estimator->trials());

$estimator->test($testing);

echo "\n";

var_dump($estimator->predict($dataset->row(1)));
