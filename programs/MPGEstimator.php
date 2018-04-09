<?php

include __DIR__ . '/../vendor/autoload.php';

use Rubix\Engine\Pipeline;
use Rubix\Engine\Prototype;
use Rubix\Engine\LeastSquares;
use Rubix\Engine\Tests\Accuracy;
use Rubix\Engine\SupervisedDataset;
use Rubix\Engine\Preprocessors\L2Normalizer;
use Rubix\Engine\Preprocessors\MissingDataImputer;
use League\Csv\Reader;

echo '╔═════════════════════════════════════════════════════╗' . "\n";
echo '║                                                     ║' . "\n";
echo '║ MPG Estimator using Least Squares                   ║' . "\n";
echo '║                                                     ║' . "\n";
echo '╚═════════════════════════════════════════════════════╝' . "\n";

echo  "\n";

$dataset = Reader::createFromPath(dirname(__DIR__) . '/datasets/auto-mpg.csv')->setDelimiter(',')->getRecords();

$dataset = SupervisedDataset::fromIterator($dataset);

list ($training, $testing) = $dataset->randomize()->split(0.3);

$prototype = new Prototype(new Pipeline(new LeastSquares(), [new MissingDataImputer('?'), new L2Normalizer()]), [new Accuracy(0.8)]);

$prototype->train($training);

$prototype->test($testing);
