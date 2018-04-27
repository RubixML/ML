<?php

include dirname(__DIR__) . '/vendor/autoload.php';

use Rubix\Engine\Ridge;
use Rubix\Engine\Pipeline;
use Rubix\Engine\Prototype;
use Rubix\Engine\GridSearch;
use Rubix\Engine\Metrics\RSquared;
use Rubix\Engine\Datasets\Supervised;
use Rubix\Engine\Reports\RegressionAnalysis;
use Rubix\Engine\Transformers\ZScaleStandardizer;
use League\Csv\Reader;

echo '╔═════════════════════════════════════════════════════╗' . "\n";
echo '║                                                     ║' . "\n";
echo '║ Wine Quality Tester using Ridge Regression          ║' . "\n";
echo '║                                                     ║' . "\n";
echo '╚═════════════════════════════════════════════════════╝' . "\n";

echo  "\n";

$dataset = Reader::createFromPath(dirname(__DIR__) . '/datasets/winequality-red.csv')->setDelimiter(';')->getRecords();

$dataset = Supervised::fromIterator($dataset);

list($training, $testing) = $dataset->randomize()->split(0.80);

$estimator = new Prototype(new Pipeline(new GridSearch(Ridge::class, [[0.0, 0.1, 0.25, 0.5, 1.0, 5.0, 10.0]], new RSquared()), [
    new ZScaleStandardizer(),
]), [
    new RegressionAnalysis(),
]);

$estimator->train($training);

$estimator->test($testing);
