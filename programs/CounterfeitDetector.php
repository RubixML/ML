<?php

include dirname(__DIR__) . '/vendor/autoload.php';

use Rubix\Engine\Pipeline;
use Rubix\Engine\Prototype;
use Rubix\Engine\Tests\MCC;
use Rubix\Engine\Tests\F1Score;
use Rubix\Engine\Tests\Accuracy;
use Rubix\Engine\SupervisedDataset;
use Rubix\Engine\Tests\Informedness;
use Rubix\Engine\MultiLayerPerceptron;
use Rubix\Engine\Persisters\Filesystem;
use Rubix\Engine\Transformers\L2Regularizer;
use Rubix\Engine\NeuralNetwork\Optimizers\Adam;
use League\Csv\Reader;

echo '╔═════════════════════════════════════════════════════╗' . "\n";
echo '║                                                     ║' . "\n";
echo '║ Counterfeit Detector using Multi Layer Perceptron   ║' . "\n";
echo '║                                                     ║' . "\n";
echo '╚═════════════════════════════════════════════════════╝' . "\n";

$dataset = Reader::createFromPath(dirname(__DIR__) . '/datasets/banknotes.csv')->setDelimiter(',')->getRecords();

$dataset = SupervisedDataset::fromIterator($dataset);

list($training, $testing) = $dataset->randomize()->stratifiedSplit(0.3);

$prototype = new Prototype(new Pipeline(new MultiLayerPerceptron($dataset->columns(), [10, 10], $dataset->labels(), 3, 10, new Adam(0.01)), [new L2Regularizer()]), [new Accuracy(), new F1Score(), new MCC(), new Informedness()]);

$prototype->train($training);

$persister = new Filesystem(dirname(__DIR__) . '/models/conterfeit_detector.model');

$persister->save($prototype);

$persister->load()->test($testing);
