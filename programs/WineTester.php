<?php

include dirname(__DIR__) . '/vendor/autoload.php';

use Rubix\Engine\Ridge;
use Rubix\Engine\Pipeline;
use Rubix\Engine\Prototype;
use Rubix\Engine\Datasets\Supervised;
use Rubix\Engine\Reports\RegressionAnalysis;
use Rubix\Engine\Transformers\MinMaxNormalizer;
use Rubix\Engine\Transformers\DensePolynomialExpander;
use League\Csv\Reader;

$alpha = $argv[1] ?? 1.0;

echo '╔═════════════════════════════════════════════════════╗' . "\n";
echo '║                                                     ║' . "\n";
echo '║ Wine Tester using a Ridge Regression                ║' . "\n";
echo '║                                                     ║' . "\n";
echo '╚═════════════════════════════════════════════════════╝' . "\n";

echo  "\n";

$dataset = Reader::createFromPath(dirname(__DIR__) . '/datasets/winequality-red.csv')->setDelimiter(';')->getRecords();

$dataset = Supervised::fromIterator($dataset);

list($training, $testing) = $dataset->randomize()->stratifiedSplit(0.8);

$estimator = new Prototype(new Pipeline(new Ridge($alpha), [
    new DensePolynomialExpander(3),
    new MinMaxNormalizer(),
]), [
    new RegressionAnalysis(),
]);

$estimator->train($training);

$estimator->test($testing);

var_dump($estimator->predict($dataset->sample(1)));
