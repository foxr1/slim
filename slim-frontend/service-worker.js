// service-worker.js

// This event listener waits for messages from the main page.
self.addEventListener('message', event => {
  // Check if the message is the one we expect for showing a notification.
  if (event.data && event.data.type === 'SHOW_NOTIFICATION') {
    const { title, options } = event.data.payload;
    // Use the service worker's registration to show the notification.
    // This is the method required by modern mobile browsers.
    self.registration.showNotification(title, options);
  }
});

// This helps the service worker take control of the page immediately.
self.addEventListener('activate', event => {
  event.waitUntil(clients.claim());
});
