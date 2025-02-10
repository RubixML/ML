<?php

declare(strict_types = 1);

namespace Rubix\ML\Tests\Loggers;

use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\Attributes\Group;
use Rubix\ML\Loggers\Screen;
use PHPUnit\Framework\TestCase;
use Psr\Log\LogLevel;

#[Group('Loggers')]
#[CoversClass(Screen::class)]
class ScreenTest extends TestCase
{
    protected Screen $logger;

    protected function setUp() : void
    {
        $this->logger = new Screen(channel: 'default');
    }

    public function testLog() : void
    {
        $this->expectOutputRegex('/\b(default.INFO: test)\b/');

        $this->logger->log(level: LogLevel::INFO, message: 'test');
    }
}
