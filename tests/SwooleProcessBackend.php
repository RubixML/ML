<?php

namespace Rubix\ML\Tests\Backends\Swoole;

require_once __DIR__.'/../vendor/autoload.php';

use Rubix\ML\Backends\Swoole\Process as SwooleProcessBackend;
use Rubix\ML\Backends\Backend;
use Rubix\ML\Backends\Tasks\Task;
use PHPUnit\Framework\TestCase;
use Swoole\Event;

$backend = new SwooleProcessBackend();

for ($i = 0; $i < 1000; ++$i) {
    $backend->enqueue(new Task(function ($input) {
        return $input * 2;
    }, [$i]));
}

$results = $backend->process();

var_dump(count($results));
var_dump($results);
