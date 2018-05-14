<?php

include dirname(__DIR__) . '/vendor/autoload.php';

use Rubix\Engine\Pipeline;
use Rubix\Engine\Prototype;
use Rubix\Engine\Metrics\MCC;
use Rubix\Engine\Datasets\Supervised;
use Rubix\Engine\NeuralNet\Layers\Input;
use Rubix\Engine\NeuralNet\Layers\Dense;
use Rubix\Engine\Reports\ConfusionMatrix;
use Rubix\Engine\NeuralNet\Optimizers\Adam;
use Rubix\Engine\Transformers\OneHotEncoder;
use Rubix\Engine\NeuralNet\Layers\Multiclass;
use Rubix\Engine\Reports\ClassificationReport;
use Rubix\Engine\Estimators\MultiLayerPerceptron;
use Rubix\Engine\NeuralNet\ActivationFunctions\PReLU;
use Rubix\Engine\NeuralNet\ActivationFunctions\Sigmoid;
use League\Csv\Reader;

echo '╔═════════════════════════════════════════════════════╗' . "\n";
echo '║                                                     ║' . "\n";
echo '║ Mushroom Classifier using Multi Layer Perceptron    ║' . "\n";
echo '║                                                     ║' . "\n";
echo '╚═════════════════════════════════════════════════════╝' . "\n";

$dataset = Reader::createFromPath(dirname(__DIR__) . '/datasets/mushrooms.csv')->setDelimiter(',')->getRecords();

$dataset = Supervised::fromIterator($dataset);

list($training, $testing) = $dataset->randomize()->stratifiedSplit(0.8);

$estimator = new Prototype(new Pipeline(new MultiLayerPerceptron([
    new Dense(10, new PReLU()),
    new Dense(10, new PReLU()),
    new Dense(10, new PReLU()),
], new Multiclass($dataset->labels()), 5, new Adam(0.001), 0.999, new MCC(), 0.2, 3, 30), [
    new OneHotEncoder(),
]), [
    new ConfusionMatrix($dataset->labels()),
    new ClassificationReport(),
]);

$estimator->train($training);

var_dump($estimator->progress());

$estimator->test($testing);

var_dump($estimator->predict($dataset->row(0)));
