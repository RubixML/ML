<?php

namespace Rubix\ML\Tests;

use PHPUnit\Framework\TestCase;

use function Rubix\ML\argmin;
use function Rubix\ML\argmax;
use function Rubix\ML\logsumexp;

class FunctionsTest extends TestCase
{
    public function test_argmin()
    {
        $value = argmin(['yes' => 0.8, 'no' => 0.2, 'maybe' => 0.0]);

        $this->assertEquals('maybe', $value);
    }

    public function test_argmax()
    {
        $value = argmax(['yes' => 0.8, 'no' => 0.2, 'maybe' => 0.0]);

        $this->assertEquals('yes', $value);
    }

    public function test_logsumexp()
    {
        $value = logsumexp([0.5, 0.4, 0.9, 1.0, 0.2, 0.9, 0.1, 0.5, 0.7]);

        $this->assertEquals(2.8194175400311074, $value);
    }
}
