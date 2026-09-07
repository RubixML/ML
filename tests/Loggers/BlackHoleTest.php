<?php

declare(strict_types = 1);

namespace Rubix\ML\Tests\Loggers;

use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\Attributes\Test;
use PHPUnit\Framework\Attributes\Group;
use Rubix\ML\Loggers\BlackHole;
use PHPUnit\Framework\TestCase;
use Psr\Log\LogLevel;

#[Group('Loggers')]
#[CoversClass(BlackHole::class)]
class BlackHoleTest extends TestCase
{
    protected BlackHole $logger;

    protected function setUp() : void
    {
        $this->logger = new BlackHole();
    }

    #[Test]
    public function logSilentlySwallowsMessages() : void
    {
        $this->expectOutputString('');

        $this->logger->log(level: LogLevel::INFO, message: 'test');

        $this->logger->info('test', ['context' => 'value']);
    }
}
