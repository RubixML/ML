<?php

namespace Rubix\Tests\CrossValidation;

use Rubix\ML\CrossValidation\MonteCarlo;
use Rubix\ML\CrossValidation\Validator;
use PHPUnit\Framework\TestCase;

class MonteCarloTest extends TestCase
{
    protected $validator;

    public function setUp()
    {
        $this->validator = new MonteCarlo(10, 0.2, false);
    }

    public function test_build_validator()
    {
        $this->assertInstanceOf(MonteCarlo::class, $this->validator);
        $this->assertInstanceOf(Validator::class, $this->validator);
    }
}
