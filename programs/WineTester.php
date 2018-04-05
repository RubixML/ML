<?php

include __DIR__ . '/../vendor/autoload.php';

use Rubix\Engine\Prototype;
use Rubix\Engine\Tests\Performance;
use Rubix\Engine\LeastSquares;
use Rubix\Engine\Tests\Accuracy;
use Rubix\Engine\SupervisedDataset;
use Rubix\Engine\Preprocessors\L1Normalizer;
use League\Csv\Reader;

echo '╔═════════════════════════════════════════════════════╗' . "\n";
echo '║                                                     ║' . "\n";
echo '║ Wine Tester using Least Squares                     ║' . "\n";
echo '║                                                     ║' . "\n";
echo '╚═════════════════════════════════════════════════════╝' . "\n";

$dataset = Reader::createFromPath(dirname(__DIR__) . '/datasets/winequality-red.csv')->setDelimiter(',')->getRecords();

$dataset = SupervisedDataset::build($dataset);

list ($training, $testing) = $dataset->randomize()->split(0.3);

$prototype = new Prototype(new LeastSquares(), [new L1Normalizer()], [new Accuracy(), new Performance()]);

echo 'Training a Least Squares ... ';

$start = microtime(true);

$prototype->train($training->samples(), $training->outcomes());

echo 'done in ' . (string) round(microtime(true) - $start, 5) . ' seconds.' . "\n";

echo  "\n";

echo 'Testing model ...' . "\n";

$prototype->test($testing->samples(), $testing->outcomes());
