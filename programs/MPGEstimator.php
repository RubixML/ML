<?php

include __DIR__ . '/../vendor/autoload.php';

use Rubix\Engine\Prototype;
use Rubix\Engine\LeastSquares;
use Rubix\Engine\Tests\Accuracy;
use Rubix\Engine\Tests\Performance;
use Rubix\Engine\SupervisedDataset;
use Rubix\Engine\Preprocessors\L1Normalizer;
use Rubix\Engine\Preprocessors\MissingDataImputer;
use League\Csv\Reader;

echo '╔═════════════════════════════════════════════════════╗' . "\n";
echo '║                                                     ║' . "\n";
echo '║ MPG Estimator using Least Squares                   ║' . "\n";
echo '║                                                     ║' . "\n";
echo '╚═════════════════════════════════════════════════════╝' . "\n";

$dataset = Reader::createFromPath(dirname(__DIR__) . '/datasets/auto-mpg.csv')->setDelimiter(',')->getRecords();

$dataset = SupervisedDataset::build($dataset);

list ($training, $testing) = $dataset->randomize()->split(0.3);

$prototype = new Prototype(new LeastSquares(), [new MissingDataImputer('?'), new L1Normalizer()], [new Accuracy(0.8), new Performance()]);

echo 'Training a Least Squares ... ';

$start = microtime(true);

$prototype->train($training);

echo 'done in ' . (string) round(microtime(true) - $start, 5) . ' seconds.' . "\n";

echo  "\n";

echo 'Testing model ...' . "\n";

$prototype->test($testing);
