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
    public function transformWithDefaultJitter() : void
    {
        $transformer = new ImageRotator(0.0);

        $dataset = Unlabeled::quick([
            [imagecreatefrompng('./tests/test.png'), 'whatever', 69],
        ]);

        $dataset->apply($transformer);

        $sample = $dataset->sample(0);

        $image = $sample[0];

        $this->assertEquals(32, imagesx($image));
        $this->assertEquals(32, imagesy($image));
        $this->assertSame('whatever', $sample[1]);
    }

    /**
     * @test
     */
    public function transformWideImage90Degrees() : void
    {
        $source = imagecreatetruecolor(80, 40);
        $dataset = Unlabeled::quick([
            [$source, 'whatever', 69],
        ]);

        foreach ([90.0, 270.0] as $degrees) {
            $mock = $this->createPartialMock(ImageRotator::class, ['rotationAngle']);
            $mock->method('rotationAngle')->will($this->returnValue($degrees));

            $dataset->apply($mock);
            $sample = $dataset->sample(0);

            $this->assertSame(80, imagesx($sample[0]));
            $this->assertSame(40, imagesy($sample[0]));
            $this->assertSame('whatever', $sample[1]);

            imagedestroy($sample[0]);
        }
    }

    /**
     * @test
     */
    public function transformTallImage90Degrees() : void
    {
        $source = imagecreatetruecolor(20, 100);
        $dataset = Unlabeled::quick([
            [$source, 'whatever', 69],
        ]);

        foreach ([90.0, 270.0] as $degrees) {
            $mock = $this->createPartialMock(ImageRotator::class, ['rotationAngle']);
            $mock->method('rotationAngle')->will($this->returnValue($degrees));

            $dataset->apply($mock);
            $sample = $dataset->sample(0);

            $this->assertSame(20, imagesx($sample[0]));
            $this->assertSame(100, imagesy($sample[0]));
            $this->assertSame('whatever', $sample[1]);

            imagedestroy($sample[0]);
        }
    }

    /**
     * @test
     */
    public function transformExtremeRatioImage90Degrees() : void
    {
        $source = imagecreatetruecolor(200, 5);
        $dataset = Unlabeled::quick([
            [$source, 'whatever', 69],
        ]);

        $mock = $this->createPartialMock(ImageRotator::class, ['rotationAngle']);
        $mock->method('rotationAngle')->will($this->returnValue(90.0));

        $dataset->apply($mock);
        $sample = $dataset->sample(0);

        $this->assertSame(200, imagesx($sample[0]));
        $this->assertSame(5, imagesy($sample[0]));
    }

    /**
     * @test
     */
    public function transformSquareImage45Degrees() : void
    {
        $source = imagecreatetruecolor(32, 32);
        $dataset = Unlabeled::quick([
            [$source, 'whatever', 69],
        ]);

        $mock = $this->createPartialMock(ImageRotator::class, ['rotationAngle']);
        $mock->method('rotationAngle')->will($this->returnValue(45.0));

        $dataset->apply($mock);
        $sample = $dataset->sample(0);

        $this->assertSame(32, imagesx($sample[0]));
        $this->assertSame(32, imagesy($sample[0]));
        $this->assertSame('whatever', $sample[1]);
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
