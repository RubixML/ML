<?php

namespace Rubix\ML\Tests\Transformers;

use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\Attributes\Group;
use PHPUnit\Framework\Attributes\RequiresPhpExtension;
use PHPUnit\Framework\TestCase;
use Rubix\ML\DataType;
use Rubix\ML\Exceptions\InvalidArgumentException;
use Rubix\ML\Transformers\ColorJitter;

use function imagecolorat;
use function imagecreatetruecolor;
use function imagefill;
use function imagesx;
use function imagesy;

#[Group('Transformers')]
#[RequiresPhpExtension('gd')]
#[CoversClass(ColorJitter::class)]
class ColorJitterTest extends TestCase
{
    /**
     * @var ColorJitter
     */
    protected ColorJitter $transformer;

    /**
     * @var resource|\GdImage
     */
    protected $image;

    protected function setUp() : void
    {
        $this->transformer = new ColorJitter(0.1, 0.1, 0.1, 0.1);

        // Create a simple test image
        $this->image = imagecreatetruecolor(10, 10);

        imagefill($this->image, 0, 0, imagecolorallocate($this->image, 100, 150, 200));
    }

    public function testConstructor() : void
    {
        $this->assertInstanceOf(ColorJitter::class, $this->transformer);
    }

    public function testConstructorWithDefaults() : void
    {
        $transformer = new ColorJitter();
        $this->assertInstanceOf(ColorJitter::class, $transformer);
    }

    public function testConstructorThrowsOnNegativeBrightness() : void
    {
        $this->expectException(InvalidArgumentException::class);
        new ColorJitter(-0.1);
    }

    public function testConstructorThrowsOnNegativeContrast() : void
    {
        $this->expectException(InvalidArgumentException::class);
        new ColorJitter(0.0, -0.1);
    }

    public function testConstructorThrowsOnNegativeSaturation() : void
    {
        $this->expectException(InvalidArgumentException::class);
        new ColorJitter(0.0, 0.0, -0.1);
    }

    public function testConstructorThrowsOnNegativeHue() : void
    {
        $this->expectException(InvalidArgumentException::class);
        new ColorJitter(0.0, 0.0, 0.0, -0.1);
    }

    public function testCompatibility() : void
    {
        $types = $this->transformer->compatibility();
        $this->assertContainsOnlyInstancesOf(DataType::class, $types);
        $this->assertEquals(DataType::all(), $types);
    }

    public function testTransformWithAllZerosUnchanged() : void
    {
        $transformer = new ColorJitter(0.0, 0.0, 0.0, 0.0, 42);
        $samples = [[$this->image]];

        $originalPx = imagecolorat($this->image, 0, 0);

        $transformer->transform($samples);

        $this->assertSame($originalPx, imagecolorat($samples[0][0], 0, 0));
        $this->assertSame(imagesx($this->image), imagesx($samples[0][0]));
        $this->assertSame(imagesy($this->image), imagesy($samples[0][0]));
    }

    public function testTransformPreservesDimensions() : void
    {
        $samples = [[$this->image]];
        $this->transformer->transform($samples);

        $this->assertSame(10, imagesx($samples[0][0]));
        $this->assertSame(10, imagesy($samples[0][0]));
    }

    public function testTransformWithMixedTypes() : void
    {
        $samples = [
            [$this->image, 5, 'string', 3.14],
        ];

        $this->transformer->transform($samples);

        $this->assertInstanceOf(\GdImage::class, $samples[0][0]);
        $this->assertSame(5, $samples[0][1]);
        $this->assertSame('string', $samples[0][2]);
        $this->assertSame(3.14, $samples[0][3]);
    }

    public function testTransformMultipleImageColumns() : void
    {
        $img2 = imagecreatetruecolor(5, 5);
        imagefill($img2, 0, 0, imagecolorallocate($img2, 10, 20, 30));

        $samples = [[$this->image, $img2, 'test']];

        $this->transformer->transform($samples);

        $this->assertInstanceOf(\GdImage::class, $samples[0][0]);
        $this->assertInstanceOf(\GdImage::class, $samples[0][1]);
        $this->assertSame('test', $samples[0][2]);

        imagedestroy($img2);
    }

    public function testTransformWithBrightness() : void
    {
        $transformer = new ColorJitter(0.3, 0.0, 0.0, 0.0, 999);
        $samples = [[$this->image]];
        $original = imagecolorat($this->image, 0, 0);

        $transformer->transform($samples);

        $jittered = imagecolorat($samples[0][0], 0, 0);
        $this->assertNotSame($original, $jittered);
    }

    public function testTransformWithContrast() : void
    {
        $transformer = new ColorJitter(0.0, 0.5, 0.0, 0.0, 123);
        $samples = [[$this->image]];
        $original = imagecolorat($this->image, 0, 0);

        $transformer->transform($samples);

        $jittered = imagecolorat($samples[0][0], 0, 0);
        $this->assertNotSame($original, $jittered);
    }

    public function testTransformWithSaturation() : void
    {
        $transformer = new ColorJitter(0.0, 0.0, 0.5, 0.0, 777);
        $samples = [[$this->image]];
        $original = imagecolorat($this->image, 0, 0);

        $transformer->transform($samples);

        $jittered = imagecolorat($samples[0][0], 0, 0);
        $this->assertNotSame($original, $jittered);
    }

    public function testTransformWithHue() : void
    {
        $transformer = new ColorJitter(0.0, 0.0, 0.0, 30.0, 666);
        $samples = [[$this->image]];
        $original = imagecolorat($this->image, 0, 0);

        $transformer->transform($samples);

        $jittered = imagecolorat($samples[0][0], 0, 0);
        $this->assertNotSame($original, $jittered);
    }

    public function testTransformWithAllParams() : void
    {
        $transformer = new ColorJitter(0.2, 0.2, 0.2, 15.0, 555);
        $samples = [[$this->image]];
        $original = imagecolorat($this->image, 0, 0);

        $transformer->transform($samples);

        $jittered = imagecolorat($samples[0][0], 0, 0);
        $this->assertNotSame($original, $jittered);
        $this->assertSame(10, imagesx($samples[0][0]));
        $this->assertSame(10, imagesy($samples[0][0]));
    }

    public function testTransformOnGrayscaleImage() : void
    {
        $gray = imagecreatetruecolor(5, 5);
        imagefill($gray, 0, 0, imagecolorallocate($gray, 128, 128, 128));
        $samples = [[$gray]];

        $this->transformer->transform($samples);

        $this->assertInstanceOf(\GdImage::class, $samples[0][0]);
        $this->assertSame(5, imagesx($samples[0][0]));
        $this->assertSame(5, imagesy($samples[0][0]));

        imagedestroy($gray);
    }

    public function testToString() : void
    {
        $this->assertSame('ColorJitter (brightness: 0.1, contrast: 0.1, saturation: 0.1, hue: 0.1)', (string) $this->transformer);
    }
}
