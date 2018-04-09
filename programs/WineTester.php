<?php

include __DIR__ . '/../vendor/autoload.php';

use Rubix\Engine\Pipeline;
use Rubix\Engine\Prototype;
use Rubix\Engine\LeastSquares;
use Rubix\Engine\Tests\Accuracy;
use Rubix\Engine\Tests\Certainty;
use Rubix\Engine\SupervisedDataset;
use Rubix\Engine\Preprocessors\L1Normalizer;
use League\Csv\Reader;

echo '╔═════════════════════════════════════════════════════╗' . "\n";
echo '║                                                     ║' . "\n";
echo '║ Wine Tester using Least Squares                     ║' . "\n";
echo '║                                                     ║' . "\n";
echo '╚═════════════════════════════════════════════════════╝' . "\n";

$dataset = Reader::createFromPath(dirname(__DIR__) . '/datasets/winequality-red.csv')->setDelimiter(',')->getRecords();

$dataset = SupervisedDataset::fromIterator($dataset);

list($training, $testing) = $dataset->randomize()->split(0.3);

$prototype = new Prototype(new Pipeline(new LeastSquares(), [new L1Normalizer()]), [new Accuracy()]);

$prototype->train($training);

$prototype->test($testing);
