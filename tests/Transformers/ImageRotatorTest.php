<?php

namespace Rubix\ML\Tests\Transformers;

use Rubix\ML\Datasets\Unlabeled;
use Rubix\ML\Transformers\ImageRotator;
use Rubix\ML\Transformers\Transformer;
use PHPUnit\Framework\TestCase;

/**
 * @group Transformers
 * @requires extension gd
 * @covers \Rubix\ML\Transformers\ImageRotator
 */
class ImageRotatorTest extends TestCase
{
    /**
     * @var ImageRotator
     */
    protected ImageRotator $transformer;

    /**
     * @before
     */
    protected function setUp() : void
    {
        $this->transformer = new ImageRotator(0.0, 1.0);
    }

    /**
     * @test
     */
    public function build() : void
    {
        $this->assertInstanceOf(ImageRotator::class, $this->transformer);
        $this->assertInstanceOf(Transformer::class, $this->transformer);
    }

    /**
     * @test
     */
    public function transform() : void
    {
        $dataset = Unlabeled::quick([
            [imagecreatefrompng('./tests/test.png'), 'whatever', 69],
        ]);

        $mock = $this->createPartialMock(ImageRotator::class, ['rotationAngle']);

        $mock->method('rotationAngle')->will($this->returnValue(-180.0));

        $dataset->apply($mock);

        $sample = $dataset->sample(0);

        ob_start();

        imagepng($sample[0]);

        $raw = ob_get_clean();

        $expected = file_get_contents('./tests/test_rotated.png');

        $this->assertEquals($expected, $raw);
    }
}
