<?php

include dirname(__DIR__) . '/vendor/autoload.php';

use Rubix\Engine\Pipeline;
use Rubix\Engine\Prototype;
use Rubix\Engine\Datasets\Supervised;
use Rubix\Engine\NeuralNet\Layers\Input;
use Rubix\Engine\Reports\ConfusionMatrix;
use Rubix\Engine\NeuralNet\Layers\Binary;
use Rubix\Engine\Estimators\RandomForest;
use Rubix\Engine\Estimators\DummyEstimator;
use Rubix\Engine\NeuralNet\Optimizers\Adam;
use Rubix\Engine\Reports\ClassificationReport;
use Rubix\Engine\Estimators\Strategies\PopularityContest;
use League\Csv\Reader;

echo '╔═════════════════════════════════════════════════════╗' . "\n";
echo '║                                                     ║' . "\n";
echo '║ Counterfeit Detector using Random Forest            ║' . "\n";
echo '║                                                     ║' . "\n";
echo '╚═════════════════════════════════════════════════════╝' . "\n";

echo "\n";

$dataset = Reader::createFromPath(dirname(__DIR__) . '/datasets/banknotes.csv')->setDelimiter(',');

$dataset = Supervised::fromIterator($dataset);

list($training, $testing) = $dataset->randomize()->stratifiedSplit(0.80);

$estimator = new Prototype(new Pipeline(new RandomForest(20, 0.5, 1, 10), [
    //
]), [
    new ConfusionMatrix($dataset->labels()),
    new ClassificationReport(),
]);

$dummy = new Prototype(new DummyEstimator(new PopularityContest()), [
    new ConfusionMatrix($dataset->labels()),
    new ClassificationReport(),
]);

$dummy->train($training);
$estimator->train($training);

$dummy->test($testing);
$estimator->test($testing);

echo "\n";

var_dump($estimator->predict($dataset->row(0)));
