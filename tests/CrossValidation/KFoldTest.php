<?php

namespace Rubix\Tests\CrossValidation;

use Rubix\ML\CrossValidation\KFold;
use PHPUnit\Framework\TestCase;

class KFoldTest extends TestCase
{
    protected $validator;

    public function setUp()
    {
        $this->validator = new KFold(10);
    }

    public function test_build_validator()
    {
        $this->assertInstanceOf(KFold::class, $this->validator);
    }
}
