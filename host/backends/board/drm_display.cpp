// DRM/KMS display (see fcw/board.hpp): primary plane = camera frame (XRGB8888), overlay plane =
// annotations (ARGB8888, alpha-blended by the DisplayPort hardware). Double-buffered dumb buffers.
#include <drm_fourcc.h>
#include <fcntl.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <xf86drm.h>
#include <xf86drmMode.h>

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cstring>
#include <filesystem>
#include <regex>
#include <stdexcept>
#include <vector>

#include "fcw/board.hpp"
#include "fcw/overlay.hpp"

namespace fcw::board {

namespace {

struct DumbBuffer {
  uint32_t handle = 0, fb = 0, pitch = 0;
  uint64_t size = 0;
  uint8_t* map = nullptr;
};

DumbBuffer createBuffer(int fd, uint32_t w, uint32_t h, uint32_t format) {
  drm_mode_create_dumb create{};
  create.width = w;
  create.height = h;
  create.bpp = 32;
  if (drmIoctl(fd, DRM_IOCTL_MODE_CREATE_DUMB, &create)) throw std::runtime_error("DRM: cannot create a dumb buffer");
  DumbBuffer b;
  b.handle = create.handle;
  b.pitch = create.pitch;
  b.size = create.size;
  const uint32_t handles[4] = {b.handle}, pitches[4] = {b.pitch}, offsets[4] = {0};
  if (drmModeAddFB2(fd, w, h, format, handles, pitches, offsets, &b.fb, 0)) throw std::runtime_error("DRM: cannot add a framebuffer");
  drm_mode_map_dumb map{};
  map.handle = b.handle;
  if (drmIoctl(fd, DRM_IOCTL_MODE_MAP_DUMB, &map)) throw std::runtime_error("DRM: cannot map a dumb buffer");
  void* p = mmap(nullptr, b.size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, map.offset);
  if (p == MAP_FAILED) throw std::runtime_error("DRM: mmap failed");
  b.map = static_cast<uint8_t*>(p);
  std::memset(b.map, 0, b.size);
  return b;
}

uint64_t planeType(int fd, uint32_t plane) {
  uint64_t type = DRM_PLANE_TYPE_OVERLAY;
  if (drmModeObjectPropertiesPtr props = drmModeObjectGetProperties(fd, plane, DRM_MODE_OBJECT_PLANE)) {
    for (uint32_t i = 0; i < props->count_props; ++i)
      if (drmModePropertyPtr p = drmModeGetProperty(fd, props->props[i])) {
        if (std::strcmp(p->name, "type") == 0) type = props->prop_values[i];
        drmModeFreeProperty(p);
      }
    drmModeFreeObjectProperties(props);
  }
  return type;
}

bool hasConnectedDisplay(const std::string& card) {
  const int fd = open(card.c_str(), O_RDWR | O_CLOEXEC);
  if (fd < 0) return false;
  bool connected = false;
  if (drmModeResPtr res = drmModeGetResources(fd)) {  // null for render-only devices (GPU, zocl)
    for (int i = 0; i < res->count_connectors && !connected; ++i)
      if (drmModeConnectorPtr c = drmModeGetConnector(fd, res->connectors[i])) {
        connected = c->connection == DRM_MODE_CONNECTED && c->count_modes > 0;
        drmModeFreeConnector(c);
      }
    drmModeFreeResources(res);
  }
  close(fd);
  return connected;
}

}  // namespace

std::string findDisplayCard(const std::string& dri) {
  std::vector<std::string> cards;
  if (std::filesystem::is_directory(dri))
    for (const auto& e : std::filesystem::directory_iterator(dri))
      if (std::regex_match(e.path().filename().string(), std::regex("card[0-9]+"))) cards.push_back(e.path().string());
  std::sort(cards.begin(), cards.end());
  for (const auto& card : cards)
    if (hasConnectedDisplay(card)) return card;
  throw std::runtime_error("DRM: no card in " + dri + " has a connected display (" + std::to_string(cards.size()) + " cards)");
}

struct DrmDisplay::Impl {
  int fd = -1;
  uint32_t crtc = 0, connector = 0, primary = 0, overlay = 0;
  drmModeModeInfo mode{};
  DumbBuffer video[2], annot[2];
  OverlayPainter painter[2];  // one per overlay buffer: each clears only what it drew there last time
  int current = 0;
};

DrmDisplay::DrmDisplay(const std::string& card) : impl_(std::make_unique<Impl>()) {
  Impl& d = *impl_;
  const std::string path = card.empty() ? findDisplayCard() : card;
  d.fd = open(path.c_str(), O_RDWR | O_CLOEXEC);
  if (d.fd < 0) throw std::runtime_error("DRM: cannot open " + path);
  drmSetClientCap(d.fd, DRM_CLIENT_CAP_UNIVERSAL_PLANES, 1);
  drmModeResPtr res = drmModeGetResources(d.fd);
  if (!res) throw std::runtime_error("DRM: no resources");
  int crtc_index = -1;
  for (int i = 0; i < res->count_connectors && !d.connector; ++i) {
    drmModeConnectorPtr c = drmModeGetConnector(d.fd, res->connectors[i]);
    if (c && c->connection == DRM_MODE_CONNECTED && c->count_modes > 0) {
      d.connector = c->connector_id;
      d.mode = c->modes[0];  // preferred mode first
      drmModeEncoderPtr e = drmModeGetEncoder(d.fd, c->encoder_id ? c->encoder_id : c->encoders[0]);
      for (int k = 0; e && k < res->count_crtcs; ++k)
        if (e->possible_crtcs & (1u << k)) { d.crtc = res->crtcs[k]; crtc_index = k; break; }
      if (e) drmModeFreeEncoder(e);
    }
    if (c) drmModeFreeConnector(c);
  }
  drmModeFreeResources(res);
  if (!d.connector || crtc_index < 0) throw std::runtime_error("DRM: no connected display");
  // Planes of this CRTC: the primary for video, an overlay that takes ARGB8888 for annotations.
  if (drmModePlaneResPtr planes = drmModeGetPlaneResources(d.fd)) {
    for (uint32_t i = 0; i < planes->count_planes; ++i) {
      drmModePlanePtr p = drmModeGetPlane(d.fd, planes->planes[i]);
      if (!p || !(p->possible_crtcs & (1u << crtc_index))) { if (p) drmModeFreePlane(p); continue; }
      bool argb = false;
      for (uint32_t f = 0; f < p->count_formats; ++f) argb |= p->formats[f] == DRM_FORMAT_ARGB8888;
      const uint64_t type = planeType(d.fd, p->plane_id);
      if (type == DRM_PLANE_TYPE_PRIMARY && !d.primary) d.primary = p->plane_id;
      else if (type == DRM_PLANE_TYPE_OVERLAY && argb && !d.overlay) d.overlay = p->plane_id;
      drmModeFreePlane(p);
    }
    drmModeFreePlaneResources(planes);
  }
  for (int k = 0; k < 2; ++k) {
    d.video[k] = createBuffer(d.fd, d.mode.hdisplay, d.mode.vdisplay, DRM_FORMAT_XRGB8888);
    if (d.overlay) d.annot[k] = createBuffer(d.fd, d.mode.hdisplay, d.mode.vdisplay, DRM_FORMAT_ARGB8888);
  }
  if (drmModeSetCrtc(d.fd, d.crtc, d.video[0].fb, 0, 0, &d.connector, 1, &d.mode))
    throw std::runtime_error("DRM: mode set failed on " + path + " (is the desktop still running? sudo xmutil desktop_disable, or use --window)");
}

DrmDisplay::~DrmDisplay() {
  if (!impl_) return;
  for (auto* set : {impl_->video, impl_->annot})
    for (int k = 0; k < 2; ++k)
      if (set[k].map) munmap(set[k].map, set[k].size);
  if (impl_->fd >= 0) close(impl_->fd);
}

void DrmDisplay::show(const FcwFrame& f, cv::Size processing) {
  Impl& d = *impl_;
  d.current ^= 1;
  const int W = d.mode.hdisplay, H = d.mode.vdisplay;
  DumbBuffer& v = d.video[d.current];
  cv::Mat screen(H, W, CV_8UC4, v.map, v.pitch);
  const cv::Rect shown(0, 0, std::min(W, f.frame.cols), std::min(H, f.frame.rows));
  f.frame(shown).copyTo(screen(shown));  // BGRx == XRGB8888 in memory
  // Annotations: on the overlay plane if there is one, else straight into the (just rewritten) video buffer.
  if (d.overlay) {
    cv::Mat annot(H, W, CV_8UC4, d.annot[d.current].map, d.annot[d.current].pitch);
    d.painter[d.current].paint(annot, f, processing, f.frame.size());
  } else {
    OverlayPainter().paint(screen, f, processing, f.frame.size());
  }
  // Flip both planes right after a vertical blank, so a scan-out never shows half of each frame.
  drmVBlank vbl{};
  vbl.request.type = DRM_VBLANK_RELATIVE;
  vbl.request.sequence = 1;
  drmWaitVBlank(d.fd, &vbl);
  drmModeSetPlane(d.fd, d.primary ? d.primary : 0, d.crtc, v.fb, 0, 0, 0, W, H, 0, 0, W << 16, H << 16);
  if (d.overlay)
    drmModeSetPlane(d.fd, d.overlay, d.crtc, d.annot[d.current].fb, 0, 0, 0, W, H, 0, 0, W << 16, H << 16);
}

}  // namespace fcw::board
