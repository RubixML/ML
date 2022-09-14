<?php

namespace Rubix\ML\Tests\Transformers;

use Rubix\ML\Datasets\Unlabeled;
use Rubix\ML\Transformers\RandomizedImageRotator;
use Rubix\ML\Transformers\Transformer;
use PHPUnit\Framework\TestCase;

/**
 * @group Transformers
 * @requires extension gd
 * @covers \Rubix\ML\Transformers\RandomizedImageRotator
 */
class RandomizedImageRotatorTest extends TestCase
{
    /**
     * @var \Rubix\ML\Datasets\Unlabeled
     */
    protected \Rubix\ML\Datasets\Unlabeled $dataset;

    /**
     * @var \Rubix\ML\Transformers\RandomizedImageRotator
     */
    protected \Rubix\ML\Transformers\RandomizedImageRotator $transformer;

    /**
     * @before
     */
    protected function setUp() : void
    {
        $this->dataset = Unlabeled::quick([
            [imagecreatefrompng('./tests/test.png'), 'whatever'],
        ]);

        $this->transformer = new RandomizedImageRotator(90.0);
    }

    /**
     * @test
     */
    public function build() : void
    {
        $this->assertInstanceOf(RandomizedImageRotator::class, $this->transformer);
        $this->assertInstanceOf(Transformer::class, $this->transformer);
    }

    /**
     * @test
     */
    public function transform() : void
    {
        $mock = $this->createPartialMock(RandomizedImageRotator::class, ['randomRotationAngle']);

        $mock->method('randomRotationAngle')->will($this->returnValue(180.0));

        $expected = file_get_contents('./tests/test_rotated.png');

        $this->dataset->apply($mock);

        $sample = $this->dataset->sample(0);

        ob_start();

        imagepng($sample[0]);

        $raw = ob_get_clean();

        $this->assertEquals($expected, $raw);
    }
}
