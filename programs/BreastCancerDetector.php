<?php

include dirname(__DIR__) . '/vendor/autoload.php';

use Rubix\Engine\Pipeline;
use Rubix\Engine\Prototype;
use Rubix\Engine\Datasets\Supervised;
use Rubix\Engine\Estimators\RandomForest;
use Rubix\Engine\Reports\ConfusionMatrix;
use Rubix\Engine\Reports\ClassificationReport;
use Rubix\Engine\Transformers\MissingDataImputer;
use League\Csv\Reader;

echo '╔═════════════════════════════════════════════════════╗' . "\n";
echo '║                                                     ║' . "\n";
echo '║ Breast Cancer Screener using Random Forest          ║' . "\n";
echo '║                                                     ║' . "\n";
echo '╚═════════════════════════════════════════════════════╝' . "\n";

echo  "\n";

$dataset = Reader::createFromPath(dirname(__DIR__) . '/datasets/breast-cancer.csv')->setDelimiter(',');

$dataset = Supervised::fromIterator($dataset);

list($training, $testing) = $dataset->randomize()->split(0.8);

$estimator = new Prototype(new Pipeline(new RandomForest(50, 0.2, 1, 10), [
    new MissingDataImputer('?'),
]), [
    new ConfusionMatrix($dataset->labels()),
    new ClassificationReport(),
]);

$estimator->train($training);

$estimator->test($testing);

echo "\n";

var_dump($estimator->predict($dataset->row(1)));
