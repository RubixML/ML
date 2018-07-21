<?php

namespace Rubix\Tests\CrossValidation;

use Rubix\ML\CrossValidation\HoldOut;
use Rubix\ML\CrossValidation\Validator;
use PHPUnit\Framework\TestCase;

class HoldOutTest extends TestCase
{
    protected $validator;

    public function setUp()
    {
        $this->validator = new HoldOut(0.2);
    }

    public function test_build_validator()
    {
        $this->assertInstanceOf(HoldOut::class, $this->validator);
        $this->assertInstanceOf(Validator::class, $this->validator);
    }
}
