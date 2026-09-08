<?php

declare(strict_types = 1);

namespace Rubix\ML\Tests\Transformers;

use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\Attributes\Test;
use PHPUnit\Framework\Attributes\Group;
use PHPUnit\Framework\Attributes\RequiresPhpExtension;
use Rubix\ML\Datasets\Unlabeled;
use Rubix\ML\Transformers\ImageRotator;
use Rubix\ML\Transformers\Transformer;
use PHPUnit\Framework\TestCase;

#[Group('Transformers')]
#[RequiresPhpExtension('gd')]
#[CoversClass(ImageRotator::class)]
class ImageRotatorTest extends TestCase
{
    protected ImageRotator $transformer;

    protected function setUp() : void
    {
        $this->transformer = new ImageRotator(offset: 0.0, jitter: 1.0);
    }

    #[Test]
    public function build() : void
    {
        $this->assertInstanceOf(ImageRotator::class, $this->transformer);
        $this->assertInstanceOf(Transformer::class, $this->transformer);
    }

    #[Test]
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

    #[Test]
    public function transformWideImage90Degrees() : void
    {
        foreach ([90.0, 270.0] as $degrees) {
            $source = imagecreatetruecolor(80, 40);
            $dataset = Unlabeled::quick([
                [$source, 'whatever', 69],
            ]);

            $mock = $this->createPartialMock(ImageRotator::class, ['rotationAngle']);
            $mock->method('rotationAngle')->will($this->returnValue($degrees));

            $dataset->apply($mock);
            $sample = $dataset->sample(0);

            $this->assertSame(80, imagesx($sample[0]));
            $this->assertSame(40, imagesy($sample[0]));
            $this->assertSame('whatever', $sample[1]);

            if ($sample[0] !== $source) {
                imagedestroy($sample[0]);
            }

            imagedestroy($source);
        }
    }

    #[Test]
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

    #[Test]
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

    #[Test]
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

    #[Test]
    public function transform() : void
    {
        $dataset = Unlabeled::quick([
            [imagecreatefrompng('./tests/test.png'), 'whatever', 69],
        ]);

        $mock = $this->createPartialMock(ImageRotator::class, ['rotationAngle']);

        $mock->expects($this->once())->method('rotationAngle')->willReturn(-180.0);

        $dataset->apply($mock);

        $sample = $dataset->sample(0);

        self::assertTrue(is_resource($sample[0]) || $sample[0] instanceof \GdImage);
        self::assertEquals(32, imagesx($sample[0]));
        self::assertEquals(32, imagesy($sample[0]));
        self::assertSame('whatever', $sample[1]);
        self::assertEquals(69, $sample[2]);
    }
}
