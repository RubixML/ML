<?php

namespace Rubix\ML\Tests\Transformers;

use Rubix\ML\Datasets\Unlabeled;
use Rubix\ML\Transformers\BooleanConverter;
use Rubix\ML\Transformers\Transformer;
use PHPUnit\Framework\TestCase;

/**
 * @group Transformers
 * @covers \Rubix\ML\Transformers\BooleanConverterTest
 */
class BooleanConverterTest extends TestCase
{
    /**
     * @var \Rubix\ML\Datasets\Unlabeled
     */
    protected $dataset;

    /**
     * @var \Rubix\ML\Transformers\BooleanConverter
     */
    protected $transformer;

    /**
     * @before
     */
    protected function setUp() : void
    {
        $this->dataset = new Unlabeled([
            [true, 'true', '1', 1],
            [false, 'false', '0', 0],
        ]);

        $this->transformer = new BooleanConverter('::true::', '::false::');
    }

    /**
     * @test
     */
    public function build() : void
    {
        $this->assertInstanceOf(BooleanConverter::class, $this->transformer);
        $this->assertInstanceOf(Transformer::class, $this->transformer);
    }

    /**
     * @test
     */
    public function transform() : void
    {
        $this->dataset->apply($this->transformer);

        $this->assertEquals([
            ['::true::', 'true', '1', 1],
            ['::false::', 'false', '0', 0],
        ], $this->dataset->samples());
    }
}
