<?php

declare(strict_types = 1);

namespace Rubix\ML\Tests\Transformers;

use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\Attributes\Group;
use PHPUnit\Framework\Attributes\RequiresPhpExtension;
use Rubix\ML\Datasets\Unlabeled;
use Rubix\ML\Transformers\ImageRotator;
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

    public function testTransform() : void
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
